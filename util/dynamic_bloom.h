// Copyright (c) 2011-present, Facebook, Inc. All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

#pragma once

#include <string>

#include "rocksdb/slice.h"

#include "port/port.h"

#include <atomic>
#include <memory>

namespace rocksdb {

class Slice;
class Allocator;
class Logger;

class DynamicBloom {
 public:
  // allocator: pass allocator to bloom filter, hence trace the usage of memory
  // total_bits: fixed total bits for the bloom
  // num_probes: number of hash probes for a single key
  // locality:  If positive, optimize for cache line locality, 0 otherwise.
  // hash_func:  customized hash function
  // huge_page_tlb_size:  if >0, try to allocate bloom bytes from huge page TLB
  //                      with this page size. Need to reserve huge pages for
  //                      it to be allocated, like:
  //                         sysctl -w vm.nr_hugepages=20
  //                     See linux doc Documentation/vm/hugetlbpage.txt
  explicit DynamicBloom(Allocator* allocator,
                        uint32_t total_bits, uint32_t locality = 0,
                        uint32_t num_probes = 6,
                        uint32_t (*hash_func)(const Slice& key) = nullptr,
                        size_t huge_page_tlb_size = 0,
                        Logger* logger = nullptr);

  explicit DynamicBloom(uint32_t num_probes = 6,
                        uint32_t (*hash_func)(const Slice& key) = nullptr);

  void SetTotalBits(Allocator* allocator, uint32_t total_bits,
                    uint32_t locality, size_t huge_page_tlb_size,
                    Logger* logger);

  ~DynamicBloom() {}

  // Assuming single threaded access to this function.
  //@NOTE not-thread-safe
  void Add(const Slice& key);

  // Like Add, but may be called concurrent with other functions.
  //@NOTE thread-safe
  void AddConcurrently(const Slice& key);

  // Assuming single threaded access to this function.
  //@NOTE not-thread-safe
  void AddHash(uint32_t hash);

  // Like AddHash, but may be called concurrent with other functions.
  //@NOTE thread-safe
  void AddHashConcurrently(uint32_t hash);

  // Multithreaded access to this function is OK
  //@NOTE thread-safe
  bool MayContain(const Slice& key) const;

  // Multithreaded access to this function is OK
  //@NOTE thread-safe
  bool MayContainHash(uint32_t hash) const;

  void Prefetch(uint32_t h);

  uint32_t GetNumBlocks() const { return kNumBlocks; }

  Slice GetRawData() const {
    return Slice(reinterpret_cast<char*>(data_), GetTotalBits() / 8);
  }

  void SetRawData(unsigned char* raw_data, uint32_t total_bits,
                  uint32_t num_blocks = 0);

  uint32_t GetTotalBits() const { return kTotalBits; }

  bool IsInitialized() const { return kNumBlocks > 0 || kTotalBits > 0; }

 private:
  uint32_t kTotalBits;
  //@NOTE 总数据量(bit)
  uint32_t kNumBlocks;
  //@NOTE 整个bit数组分成kNumBlocks块
  const uint32_t kNumProbes;
  //@NOTE hash探查次数

  uint32_t (*hash_func_)(const Slice& key);
  std::atomic<uint8_t>* data_;

  // or_func(ptr, mask) should effect *ptr |= mask with the appropriate
  // concurrency safety, working with bytes.
  template <typename OrFunc>
  void AddHash(uint32_t hash, const OrFunc& or_func);
};

inline void DynamicBloom::Add(const Slice& key) { AddHash(hash_func_(key)); }

inline void DynamicBloom::AddConcurrently(const Slice& key) {
  AddHashConcurrently(hash_func_(key));
}

inline void DynamicBloom::AddHash(uint32_t hash) {
  AddHash(hash, [](std::atomic<uint8_t>* ptr, uint8_t mask) {
    ptr->store(ptr->load(std::memory_order_relaxed) | mask,
               std::memory_order_relaxed);
  });
  //@NOTE 非原子操作，not thread safe
  //为啥不直接用 fetch_or? 既然已经约定此函数not thread safe, 
  //直接用fetch_or只需要1次调用，难道fetch_or的代价比load+or+store更高？
  //fetch_or比一般的单个load,or,store指令占用更多cpu时钟，
  //从微观上看可能使得cpu流水线受阻，而load,or,store分成3个操作更好地利用cpu指令流水线
  //大概是这样的，不过我很怀疑这样做是否能达到期望的效果？
  //而直接用fetch_or不一定效果更差？
}

inline void DynamicBloom::AddHashConcurrently(uint32_t hash) {
  AddHash(hash, [](std::atomic<uint8_t>* ptr, uint8_t mask) {
    // Happens-before between AddHash and MaybeContains is handled by
    // access to versions_->LastSequence(), so all we have to do here is
    // avoid races (so we don't give the compiler a license to mess up
    // our code) and not lose bits.  std::memory_order_relaxed is enough
    // for that.
    if ((mask & ptr->load(std::memory_order_relaxed)) != mask) {
      ptr->fetch_or(mask, std::memory_order_relaxed);
    }
  });
  //@NOTE 没看懂和前一个AddHash有啥区别? AddHashConcurrent需要4个指令，代价比AddHash高
  //为啥要做一次判断？这样可以减少多线程执行fetch_or的竞争等待
}

inline bool DynamicBloom::MayContain(const Slice& key) const {
  return (MayContainHash(hash_func_(key)));
}

inline void DynamicBloom::Prefetch(uint32_t h) {
  if (kNumBlocks != 0) {
    uint32_t b = ((h >> 11 | (h << 21)) % kNumBlocks) * (CACHE_LINE_SIZE * 8);
    PREFETCH(&(data_[b]), 0, 3);
    //@NOTE BUG 这个PREFERCH的位置可能不对？
    //b表示bit位置序号，所以在预取时应该访问data_[b/8]
  }
}

inline bool DynamicBloom::MayContainHash(uint32_t h) const {
  assert(IsInitialized());
  const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
  if (kNumBlocks != 0) {
    uint32_t b = ((h >> 11 | (h << 21)) % kNumBlocks) * (CACHE_LINE_SIZE * 8);
    for (uint32_t i = 0; i < kNumProbes; ++i) {
      // Since CACHE_LINE_SIZE is defined as 2^n, this line will be optimized
      //  to a simple and operation by compiler.
      const uint32_t bitpos = b + (h % (CACHE_LINE_SIZE * 8));
      uint8_t byteval = data_[bitpos / 8].load(std::memory_order_relaxed);
      if ((byteval & (1 << (bitpos % 8))) == 0) {
        return false;
      }
      // Rotate h so that we don't reuse the same bytes.
      h = h / (CACHE_LINE_SIZE * 8) +
          (h % (CACHE_LINE_SIZE * 8)) * (0x20000000U / CACHE_LINE_SIZE);
      //@NOTE 设CACHE_LINE_SIZE*8=2^x
      //h = h >> x | (h&((1<<x)-1))<<x
      //即h的低x位与高32-x位做bit旋转
      h += delta;
    }
  } else {
    for (uint32_t i = 0; i < kNumProbes; ++i) {
      const uint32_t bitpos = h % kTotalBits;
      uint8_t byteval = data_[bitpos / 8].load(std::memory_order_relaxed);
      if ((byteval & (1 << (bitpos % 8))) == 0) {
        return false;
      }
      h += delta;
    }
  }
  return true;
}

template <typename OrFunc>
inline void DynamicBloom::AddHash(uint32_t h, const OrFunc& or_func) {
  assert(IsInitialized());
  const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
  if (kNumBlocks != 0) {
    uint32_t b = ((h >> 11 | (h << 21)) % kNumBlocks) * (CACHE_LINE_SIZE * 8);
    for (uint32_t i = 0; i < kNumProbes; ++i) {
      // Since CACHE_LINE_SIZE is defined as 2^n, this line will be optimized
      // to a simple and operation by compiler.
      const uint32_t bitpos = b + (h % (CACHE_LINE_SIZE * 8));
      or_func(&data_[bitpos / 8], (1 << (bitpos % 8)));
      // Rotate h so that we don't reuse the same bytes.
      h = h / (CACHE_LINE_SIZE * 8) +
          (h % (CACHE_LINE_SIZE * 8)) * (0x20000000U / CACHE_LINE_SIZE);
      h += delta;
    }
  } else {
    for (uint32_t i = 0; i < kNumProbes; ++i) {
      const uint32_t bitpos = h % kTotalBits;
      or_func(&data_[bitpos / 8], (1 << (bitpos % 8)));
      h += delta;
    }
  }
}

}  // rocksdb
