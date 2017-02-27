//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under the BSD-style license found in the
//  LICENSE file in the root directory of this source tree. An additional grant
//  of patent rights can be found in the PATENTS file in the same directory.
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#pragma once
#include <atomic>
#include <memory>
#include <utility>
#include "port/likely.h"
#include "util/allocator.h"
#include "util/arena.h"
#include "util/mutexlock.h"
#include "util/thread_local.h"

// Only generate field unused warning for padding array, or build under
// GCC 4.8.1 will fail.
#ifdef __clang__
#define ROCKSDB_FIELD_UNUSED __attribute__((__unused__))
#else
#define ROCKSDB_FIELD_UNUSED
#endif  // __clang__

namespace rocksdb {

class Logger;

// ConcurrentArena wraps an Arena.  It makes it thread safe using a fast
// inlined spinlock, and adds small per-core allocation caches to avoid
// contention for small allocations.  To avoid any memory waste from the
// per-core shards, they are kept small, they are lazily instantiated
// only if ConcurrentArena actually notices concurrent use, and they
// adjust their size so that there is no fragmentation waste when the
// shard blocks are allocated from the underlying main arena.
class ConcurrentArena : public Allocator {
 public:
  // block_size and huge_page_size are the same as for Arena (and are
  // in fact just passed to the constructor of arena_.  The core-local
  // shards compute their shard_block_size as a fraction of block_size
  // that varies according to the hardware concurrency level.
  //@NOTE shard数量取最小的一个不小于8且不小于cpu核数的2的幂。
  explicit ConcurrentArena(size_t block_size = Arena::kMinBlockSize,
                           size_t huge_page_size = 0);

  char* Allocate(size_t bytes) override {
    return AllocateImpl(bytes, false /*force_arena*/,
                        [=]() { return arena_.Allocate(bytes); });
  }

  char* AllocateAligned(size_t bytes, size_t huge_page_size = 0,
                        Logger* logger = nullptr) override {
    size_t rounded_up = ((bytes - 1) | (sizeof(void*) - 1)) + 1;
    //@NOTE 按指针size对齐bytes
    assert(rounded_up >= bytes && rounded_up < bytes + sizeof(void*) &&
           (rounded_up % sizeof(void*)) == 0);

    return AllocateImpl(rounded_up, huge_page_size != 0 /*force_arena*/, [=]() {
      return arena_.AllocateAligned(rounded_up, huge_page_size, logger);
    });
  }

  size_t ApproximateMemoryUsage() const {
    //@NOTE 近似内存使用量
    std::unique_lock<SpinMutex> lock(arena_mutex_, std::defer_lock);
    if (index_mask_ != 0) {
      lock.lock();
    }
    return arena_.ApproximateMemoryUsage() - ShardAllocatedAndUnused();
  }

  size_t MemoryAllocatedBytes() const {
    //@NOTE 已申请的内存量
    return memory_allocated_bytes_.load(std::memory_order_relaxed);
  }

  size_t AllocatedAndUnused() const {
    return arena_allocated_and_unused_.load(std::memory_order_relaxed) +
           ShardAllocatedAndUnused();
  }

  size_t IrregularBlockNum() const {
    return irregular_block_num_.load(std::memory_order_relaxed);
  }

  size_t BlockSize() const override { return arena_.BlockSize(); }

 private:
  struct Shard {
    char padding[40] ROCKSDB_FIELD_UNUSED;
    mutable SpinMutex mutex;
    char* free_begin_;
    std::atomic<size_t> allocated_and_unused_;

    Shard() : allocated_and_unused_(0) {}
  };

#ifdef ROCKSDB_SUPPORT_THREAD_LOCAL
  static __thread uint32_t tls_cpuid;
#else
  enum ZeroFirstEnum : uint32_t { tls_cpuid = 0 };
#endif
//@NOTE tls Thread-local storage (TLS)
// tls_cpuid 当前tls关联的cpuid
// 用0和非0表示是否进行过repick
// https://en.wikipedia.org/wiki/Thread-local_storage

  char padding0[56] ROCKSDB_FIELD_UNUSED;

  size_t shard_block_size_;

  // shards_[i & index_mask_] is valid
  size_t index_mask_;
  std::unique_ptr<Shard[]> shards_;

  Arena arena_;
  mutable SpinMutex arena_mutex_;
  std::atomic<size_t> arena_allocated_and_unused_;
  std::atomic<size_t> memory_allocated_bytes_;
  std::atomic<size_t> irregular_block_num_;

  char padding1[56] ROCKSDB_FIELD_UNUSED;

  Shard* Repick();

  size_t ShardAllocatedAndUnused() const {
    size_t total = 0;
    for (size_t i = 0; i <= index_mask_; ++i) {
      total += shards_[i].allocated_and_unused_.load(std::memory_order_relaxed);
    }
    return total;
  }

  template <typename Func>
  char* AllocateImpl(size_t bytes, bool force_arena, const Func& func) {
    //@NOTE func 是arena申请内存的函数调用。
    uint32_t cpu;

    // Go directly to the arena if the allocation is too large, or if
    // we've never needed to Repick() and the arena mutex is available
    // with no waiting.  This keeps the fragmentation penalty of
    // concurrency zero unless it might actually confer an advantage.
    std::unique_lock<SpinMutex> arena_lock(arena_mutex_, std::defer_lock);
    if (bytes > shard_block_size_ / 4 || force_arena ||
        ((cpu = tls_cpuid) == 0 &&
         !shards_[0].allocated_and_unused_.load(std::memory_order_relaxed) &&
         arena_lock.try_lock())) {
      //@NOTE 申请内存较大 或 强制arena申请 或 核心shard无可用内存且不需要等锁
      if (!arena_lock.owns_lock()) {
        arena_lock.lock();
      }
      auto rv = func();
      Fixup();
      return rv;
    }

    // pick a shard from which to allocate
    Shard* s = &shards_[cpu & index_mask_];
    if (!s->mutex.try_lock()) {
      s = Repick();
      s->mutex.lock();
    }
    //@NOTE 优先从Shard[0]分配，若Shard[0]被锁则选取当前cpu核标号的Shard
    std::unique_lock<SpinMutex> lock(s->mutex, std::adopt_lock);

    size_t avail = s->allocated_and_unused_.load(std::memory_order_relaxed);
    if (avail < bytes) {
      // reload
      std::lock_guard<SpinMutex> reload_lock(arena_mutex_);

      // If the arena's current block is within a factor of 2 of the right
      // size, we adjust our request to avoid arena waste.
      auto exact = arena_allocated_and_unused_.load(std::memory_order_relaxed);
      assert(exact == arena_.AllocatedAndUnused());
      avail = exact >= shard_block_size_ / 2 && exact < shard_block_size_ * 2
                  ? exact
                  : shard_block_size_;
      s->free_begin_ = arena_.AllocateAligned(avail);
      //@NOTE Shard内存不够用，从arena申请
      //若arena可用内存大于shard_block_size的二分之一小于shard_block_size两倍，
      //则将可用内存全部分给Shard。
      //否则按shard_block_size分给Shard。
      //
      //s->free_begin_是否可能为nullptr?
      Fixup();
    }
    s->allocated_and_unused_.store(avail - bytes, std::memory_order_relaxed);

    char* rv;
    if ((bytes % sizeof(void*)) == 0) {
      // aligned allocation from the beginning
      rv = s->free_begin_;
      s->free_begin_ += bytes;
    } else {
      // unaligned from the end
      rv = s->free_begin_ + avail - bytes;
    }
    return rv;
  }

  void Fixup() {
    arena_allocated_and_unused_.store(arena_.AllocatedAndUnused(),
                                      std::memory_order_relaxed);
    memory_allocated_bytes_.store(arena_.MemoryAllocatedBytes(),
                                  std::memory_order_relaxed);
    irregular_block_num_.store(arena_.IrregularBlockNum(),
                               std::memory_order_relaxed);
  }

  ConcurrentArena(const ConcurrentArena&) = delete;
  ConcurrentArena& operator=(const ConcurrentArena&) = delete;
};

}  // namespace rocksdb
