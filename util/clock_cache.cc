//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under the BSD-style license found in the
//  LICENSE file in the root directory of this source tree. An additional grant
//  of patent rights can be found in the PATENTS file in the same directory.
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "util/clock_cache.h"

#ifndef SUPPORT_CLOCK_CACHE

namespace rocksdb {

std::shared_ptr<Cache> NewClockCache(size_t capacity, int num_shard_bits,
                                     bool strict_capacity_limit) {
  // Clock cache not supported.
  return nullptr;
}

}  // namespace rocksdb

#else

#include <assert.h>
#include <atomic>
#include <deque>
#include <limits>

#include "tbb/concurrent_hash_map.h"

#include "port/port.h"
#include "util/autovector.h"
#include "util/mutexlock.h"
#include "util/sharded_cache.h"

namespace rocksdb {

namespace {
//@NOTE Clock Cache算法思想：将所有的缓存单元组成一个环形链表，维护一个head
//指针指向最近一个检查过的缓存单元。资源回收从当前的head指针开始，如果缓存项
//在上一轮检查后被访问过，那么在新一轮检查中将不会被回收。
//与LRU不同，查找过程不会修改内部数据结构，每个缓存单元仅涉及使用标记bit位
//至多一次修改，因此可以获得更好的并发性能。
//
//缓存单元被使用者释放时，我们不会直接删除，而是把句柄放入回收池等待重用。
//这样可以避免内存的频繁申请释放，因为在并发场景下，这种情况一旦发生将很难应付。
//使用了tbb::concurrent_hash_map 因其支持并发erase操作。

// An implementation of the Cache interface based on CLOCK algorithm, with
// better concurrent performance than LRUCache. The idea of CLOCK algorithm
// is to maintain all cache entries in a circular list, and an iterator
// (the "head") pointing to the last examined entry. Eviction starts from the
// current head. Each entry is given a second chance before eviction, if it
// has been access since last examine. In contrast to LRU, no modification
// to the internal data-structure (except for flipping the usage bit) needs
// to be done upon lookup. This gives us opportunity to implement a cache
// with better concurrency.
//
// Each cache entry is represented by a cache handle, and all the handles
// are arranged in a circular list, as describe above. Upon erase of an entry,
// we never remove the handle. Instead, the handle is put into a recycle bin
// to be re-use. This is to avoid memory deallocation, which is hard to deal
// with in concurrent environment.
//
// The cache also maintains a concurrent hash map for lookup. Any concurrent
// hash map implementation should do the work. We currently use
// tbb::concurrent_hash_map because it supports concurrent erase.
//
// Each cache handle has the following flags and counters, which are squeezed
// in an atomic interger, to make sure the handle always be in a consistent
// state:
//
//   * In-cache bit: whether the entry is reference by the cache itself. If
//     an entry is in cache, its key would also be available in the hash map.
//   * Usage bit: whether the entry has been access by user since last
//     examine for eviction. Can be reset by eviction.
//   * Reference count: reference count by user.
//
// An entry can be reference only when it's in cache. An entry can be evicted
// only when it is in cache, has no usage since last examine, and reference
// count is zero.
//
// The follow figure shows a possible layout of the cache. Boxes represents
// cache handles and numbers in each box being in-cache bit, usage bit and
// reference count respectively.
//
//    hash map:
//      +-------+--------+
//      |  key  | handle |
//      +-------+--------+
//      | "foo" |    5   |-------------------------------------+
//      +-------+--------+                                     |
//      | "bar" |    2   |--+                                  |
//      +-------+--------+  |                                  |
//                          |                                  |
//                     head |                                  |
//                       |  |                                  |
//    circular list:     |  |                                  |
//         +-------+   +-------+   +-------+   +-------+   +-------+   +-------
//         |(0,0,0)|---|(1,1,0)|---|(0,0,0)|---|(0,1,3)|---|(1,0,0)|---|  ...
//         +-------+   +-------+   +-------+   +-------+   +-------+   +-------
//             |                       |
//             +-------+   +-----------+
//                     |   |
//                   +---+---+
//    recycle bin:   | 1 | 3 |
//                   +---+---+
//
// Suppose we try to insert "baz" into the cache at this point and the cache is
// full. The cache will first look for entries to evict, starting from where
// head points to (the second entry). It resets usage bit of the second entry,
// skips the third and fourth entry since they are not in cache, and finally
// evict the fifth entry ("foo"). It looks at recycle bin for available handle,
// grabs handle 3, and insert the key into the handle. The following figure
// shows the resulting layout.
//
//    hash map:
//      +-------+--------+
//      |  key  | handle |
//      +-------+--------+
//      | "baz" |    3   |-------------+
//      +-------+--------+             |
//      | "bar" |    2   |--+          |
//      +-------+--------+  |          |
//                          |          |
//                          |          |                                 head
//                          |          |                                   |
//    circular list:        |          |                                   |
//         +-------+   +-------+   +-------+   +-------+   +-------+   +-------
//         |(0,0,0)|---|(1,0,0)|---|(1,0,0)|---|(0,1,3)|---|(0,0,0)|---|  ...
//         +-------+   +-------+   +-------+   +-------+   +-------+   +-------
//             |                                               |
//             +-------+   +-----------------------------------+
//                     |   |
//                   +---+---+
//    recycle bin:   | 1 | 5 |
//                   +---+---+
//
//@NOTE 为什么把handle5放进回收池，选用handle3? 不直接把数据存到handle5？
//有什么危险吗？这样处理更有利于并发性能？
//
// A global mutex guards the circular list, the head, and the recycle bin.
// We additionally require that modifying the hash map needs to hold the mutex.
// As such, Modifying the cache (such as Insert() and Erase()) require to
// hold the mutex. Lookup() only access the hash map and the flags associated
// with each handle, and don't require explicit locking. Release() has to
// acquire the mutex only when it releases the last reference to the entry and
// the entry has been erased from cache explicitly. A future improvement could
// be to remove the mutex completely.
//
//@NOTE 循环链表、头指针、回收池共享一个全局锁，
//插入、删除操作需要拿到全局锁，而查询只需要访问哈希容器和每个缓存单元的usage bit。
//释放操作仅当缓存单元的最后一个外部引用且该缓存项已经被显式地从缓存删除时需要拿到锁。
//未来的改进也许可以完全抛弃互斥锁。
//
// Benchmark:
// We run readrandom db_bench on a test DB of size 13GB, with size of each
// level:
//
//    Level    Files   Size(MB)
//    -------------------------
//      L0        1       0.01
//      L1       18      17.32
//      L2      230     182.94
//      L3     1186    1833.63
//      L4     4602    8140.30
//
// We test with both 32 and 16 read threads, with 2GB cache size (the whole DB
// doesn't fits in) and 64GB cache size (the whole DB can fit in cache), and
// whether to put index and filter blocks in block cache. The benchmark runs
// with RocksDB 4.10. We got the following result:
//
// Threads Cache     Cache               ClockCache               LRUCache
//         Size  Index/Filter Throughput(MB/s)   Hit Throughput(MB/s)    Hit
//     32   2GB       yes               466.7  85.9%           433.7   86.5%
//     32   2GB       no                529.9  72.7%           532.7   73.9%
//     32  64GB       yes               649.9  99.9%           507.9   99.9%
//     32  64GB       no                740.4  99.9%           662.8   99.9%
//     16   2GB       yes               278.4  85.9%           283.4   86.5%
//     16   2GB       no                318.6  72.7%           335.8   73.9%
//     16  64GB       yes               391.9  99.9%           353.3   99.9%
//     16  64GB       no                433.8  99.8%           419.4   99.8%
//
//@NOTE ClockCache性能比LRUCache提高了不少，数据为证...
//whether to put index and filter blocks in block cache
//index, filter和block cache是什么东东？

// Cache entry meta data.
struct CacheHandle {
  Slice key;
  uint32_t hash;
  void* value;
  size_t charge;
  //@NOTE charge是什么意思？
  //charge应该是value_size，表示缓存数据量
  void (*deleter)(const Slice&, void* value);

  // Flags and counters associated with the cache handle:
  //   lowest bit: in-cache bit
  //   second lowest bit: usage bit
  //   the rest bits: reference count
  // The handle is unused when flags equals to 0. The thread decreases the count
  // to 0 is responsible to put the handle back to recycle_ and cleanup memory.
  std::atomic<uint32_t> flags;

  CacheHandle() = default;

  CacheHandle(const CacheHandle& a) { *this = a; }
  //@NOTE 是否可以考虑用std::move? 
  //CacheHandle对象适合被多于1个的变量引用吗？
  //C++11 move constructor http://en.cppreference.com/w/cpp/language/move_constructor
  // http://en.cppreference.com/w/cpp/utility/move

  CacheHandle(const Slice& k, void* v,
              void (*del)(const Slice& key, void* value))
      : key(k), value(v), deleter(del) {}
  //@NOTE 为什么charge, hash不进行初始化? 
  //有必要这么省吗?

  CacheHandle& operator=(const CacheHandle& a) {
    // Only copy members needed for deletion.
    //@NOTE 为什么？
    //可能作者假定被赋值的对象一定不在且不会被放入Cache中
    //赋值方法只有即将被删除的CacheHandle会用到
    //这样不太好，有必要这么省吗？
    key = a.key;
    value = a.value;
    deleter = a.deleter;
    return *this;
  }
};

// Key of hash map. We store hash value with the key for convenience.
struct CacheKey {
  Slice key;
  uint32_t hash_value;

  CacheKey() = default;

  CacheKey(const Slice& k, uint32_t h) {
    key = k;
    hash_value = h;
  }

  static bool equal(const CacheKey& a, const CacheKey& b) {
    return a.hash_value == b.hash_value && a.key == b.key;
  }

  static size_t hash(const CacheKey& a) {
    return static_cast<size_t>(a.hash_value);
  }
};

struct CleanupContext {
  // List of values to be deleted, along with the key and deleter.
  autovector<CacheHandle> to_delete_value;

  // List of keys to be deleted.
  autovector<const char*> to_delete_key;
  //@NOTE 为什么是const char* 而不是CacheKey?
};

// A cache shard which maintains its own CLOCK cache.
class ClockCacheShard : public CacheShard {
 public:
  // Hash map type.
  typedef tbb::concurrent_hash_map<CacheKey, CacheHandle*, CacheKey> HashTable;

  ClockCacheShard();
  ~ClockCacheShard();

  // Interfaces
  virtual void SetCapacity(size_t capacity) override;
  virtual void SetStrictCapacityLimit(bool strict_capacity_limit) override;
  virtual Status Insert(const Slice& key, uint32_t hash, void* value,
                        size_t charge,
                        void (*deleter)(const Slice& key, void* value),
                        Cache::Handle** handle,
                        Cache::Priority priority) override;
  virtual Cache::Handle* Lookup(const Slice& key, uint32_t hash) override;
  // If the entry in in cache, increase reference count and return true.
  // Return false otherwise.
  //
  // Not necessary to hold mutex_ before being called.
  virtual bool Ref(Cache::Handle* handle) override;
  //@NOTE 增加引用
  virtual void Release(Cache::Handle* handle) override;
  //@NOTE Release外部使用的释放资源接口，UnRef是私有方法
  virtual void Erase(const Slice& key, uint32_t hash) override;
  virtual size_t GetUsage() const override;
  virtual size_t GetPinnedUsage() const override;
  virtual void EraseUnRefEntries() override;
  //@NOTE 将所有已缓存资源移出缓存池，将无人认领的资源释放
  virtual void ApplyToAllCacheEntries(void (*callback)(void*, size_t),
                                      bool thread_safe) override;

  //@NOTE SetCapacity, Insert, Lookup, Release, Erease, EraseUnRefEntries会触发RecycleHandle

 private:
  static const uint32_t kInCacheBit = 1;
  //@NOTE 即2^0 , 1<<0
  static const uint32_t kUsageBit = 2;
  //@NOTE 即2^1 , 1<<1
  static const uint32_t kRefsOffset = 2;
  //@NOTE 从低位数第2个bit开始
  static const uint32_t kOneRef = 1 << kRefsOffset;
  //@NOTE 引用数量从0变到1之后的值

  // Helper functions to extract cache handle flags and counters.
  static bool InCache(uint32_t flags) { return flags & kInCacheBit; }
  static bool HasUsage(uint32_t flags) { return flags & kUsageBit; }
  static uint32_t CountRefs(uint32_t flags) { return flags >> kRefsOffset; }

  // Decrease reference count of the entry. If this decreases the count to 0,
  // recycle the entry. If set_usage is true, also set the usage bit.
  //
  // Not necessary to hold mutex_ before being called.
  void Unref(CacheHandle* handle, bool set_usage, CleanupContext* context);

  // Unset in-cache bit of the entry. Recycle the handle if necessary.
  //
  // Has to hold mutex_ before being called.
  void UnsetInCache(CacheHandle* handle, CleanupContext* context);

  // Put the handle back to recycle_ list, and put the value associated with
  // it into to-be-deleted list. It doesn't cleanup the key as it might be
  // reused by another handle.
  //
  // Has to hold mutex_ before being called.
  void RecycleHandle(CacheHandle* handle, CleanupContext* context);

  // Delete keys and values in to-be-deleted list. Call the method without
  // holding mutex, as destructors can be expensive.
  void Cleanup(const CleanupContext& context);
  //@NOTE destructors of class lock_guard?

  // Examine the handle for eviction. If the handle is in cache, usage bit is
  // not set, and referece count is 0, evict it from cache. Otherwise unset
  // the usage bit.
  //
  // Has to hold mutex_ before being called.
  bool TryEvict(CacheHandle* value, CleanupContext* context);

  // Scan through the circular list, evict entries until we get enough capacity
  // for new cache entry of specific size. Return true if success, false
  // otherwise.
  //
  // Has to hold mutex_ before being called.
  bool EvictFromCache(size_t charge, CleanupContext* context);
  //@NOTE 当缓存容量配置变化、插入操作遇到缓存填满时，触发缓存回收操作。

  CacheHandle* Insert(const Slice& key, uint32_t hash, void* value,
                      size_t change,
                      void (*deleter)(const Slice& key, void* value),
                      bool hold_reference, CleanupContext* context);

  // Guards list_, head_, and recycle_. In addition, updating table_ also has
  // to hold the mutex, to avoid the cache being in inconsistent state.
  mutable port::Mutex mutex_;

  // The circular list of cache handles. Initially the list is empty. Once a
  // handle is needed by insertion, and no more handles are available in
  // recycle bin, one more handle is appended to the end.
  //
  // We use std::deque for the circular list because we want to make sure
  // pointers to handles are valid through out the life-cycle of the cache
  // (in contrast to std::vector), and be able to grow the list (in contrast
  // to statically allocated arrays).
  std::deque<CacheHandle> list_;

  // Pointer to the next handle in the circular list to be examine for
  // eviction.
  size_t head_;

  // Recycle bin of cache handles.
  autovector<CacheHandle*> recycle_;

  // Maximum cache size.
  std::atomic<size_t> capacity_;
  //@NOTE byte

  // Current total size of the cache.
  std::atomic<size_t> usage_;
  //@NOTE 正在使用的全部缓存size, byte

  // Total un-released cache size.
  std::atomic<size_t> pinned_usage_;
  //@NOTE 正在使用的未释放的缓存size, byte

  // Whether allow insert into cache if cache is full.
  std::atomic<bool> strict_capacity_limit_;

  //@NOTE  std::atomic C++11的原子类型模板
  // http://zh.cppreference.com/w/cpp/atomic/atomic
  // http://www.cnblogs.com/haippy/p/3301408.html

  // Hash table (tbb::concurrent_hash_map) for lookup.
  HashTable table_;
};

ClockCacheShard::ClockCacheShard()
    : head_(0), usage_(0), pinned_usage_(0), strict_capacity_limit_(false) {}

ClockCacheShard::~ClockCacheShard() {
  for (auto& handle : list_) {
    uint32_t flags = handle.flags.load(std::memory_order_relaxed);
    //@NOTE std::atomic::load原子获取变量值
    //memory_order 多个线程同一瞬间访问变量的排队顺序
    //memory_order_relaxed是enum std::memory_order的一个值，
    //含义是不和其他线程进行同步或排序，只要保证此次访问原子性就行。
    //...
    // 意思是一个线程内的多个指令在cpu上并发执行，如果相邻两个指令之间没有因果关系，
    // 则后一条指令可以先于前一条指令执行，即编译器指令重排、CPU流水线指令乱序执行。
    // 而多个线程之间的对同一原子变量的访问顺序为relaxed时，就是没有顺序。。。
    // http://en.cppreference.com/w/cpp/atomic/memory_order#Relaxed_ordering
    // http://stackoverflow.com/questions/27086838/relaxed-ordering-of-c11-memory-model
    // http://blog.csdn.net/aitangyong/article/details/40550153
    // https://www.zhihu.com/question/24301047
    if (InCache(flags) || CountRefs(flags) > 0) {
      (*handle.deleter)(handle.key, handle.value);
      delete[] handle.key.data();
    }
    //@NOTE 放在CacheHandle的析构函数里是不是比较好？
  }
}

size_t ClockCacheShard::GetUsage() const {
  return usage_.load(std::memory_order_relaxed);
}

size_t ClockCacheShard::GetPinnedUsage() const {
  return pinned_usage_.load(std::memory_order_relaxed);
}

void ClockCacheShard::ApplyToAllCacheEntries(void (*callback)(void*, size_t),
                                             bool thread_safe) {
  if (thread_safe) {
    mutex_.Lock();
  }
  for (auto& handle : list_) {
    // Use relaxed semantics instead of acquire semantics since we are either
    // holding mutex, or don't have thread safe requirement.
    uint32_t flags = handle.flags.load(std::memory_order_relaxed);
    if (InCache(flags)) {
      callback(handle.value, handle.charge);
    }
    //@NOTE InCache判断为true前，handle可能从Cache中拿掉。
    //@NOTE InCache判断为true后，handle.value是不是有可能被其他线程释放掉？
    //不会因为回收一个handle需要拿到mutex_
  }
  if (thread_safe) {
    mutex_.Unlock();
  }
}

void ClockCacheShard::RecycleHandle(CacheHandle* handle,
                                    CleanupContext* context) {
  mutex_.AssertHeld();
  assert(!InCache(handle->flags) && CountRefs(handle->flags) == 0);
  context->to_delete_key.push_back(handle->key.data());
  context->to_delete_value.emplace_back(*handle);
  //@NOTE std::vector::emplace_back 在函数尾部直接构造，比push_back减少一次拷贝
  // http://zh.cppreference.com/w/cpp/container/vector/emplace_back
  handle->key.clear();
  handle->value = nullptr;
  handle->deleter = nullptr;
  recycle_.push_back(handle);
  usage_.fetch_sub(handle->charge, std::memory_order_relaxed);
}

void ClockCacheShard::Cleanup(const CleanupContext& context) {
  for (const CacheHandle& handle : context.to_delete_value) {
    if (handle.deleter) {
      (*handle.deleter)(handle.key, handle.value);
    }
  }
  for (const char* key : context.to_delete_key) {
    delete[] key;
  }
}

bool ClockCacheShard::Ref(Cache::Handle* h) {
  auto handle = reinterpret_cast<CacheHandle*>(h);
  // CAS loop to increase reference count.
  uint32_t flags = handle->flags.load(std::memory_order_relaxed);
  while (InCache(flags)) {
    // Use acquire semantics on success, as further operations on the cache
    // entry has to be order after reference count is increased.
    if (handle->flags.compare_exchange_weak(flags, flags + kOneRef,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
      if (CountRefs(flags) == 0) {
        // No reference count before the operation.
        pinned_usage_.fetch_add(handle->charge, std::memory_order_relaxed);
        //@NOTE pinned_usage_, handel->charge分别是什么含义？
        //pinned_usage_ 表示在当前容器中被外界使用、引用的缓存size总和
        //usage_ 表示在当前容器中所有缓存size总和
        //charge 表示CacheHandle的数据量
      }
      return true;
    }
    //@NOTE std::atomic::compare_exchange_weak 与第一个参数比较，若相等则交换
    //Return true,  *this与第一个参数相等，将第二个参数写入*this
    //Return false, 第一个参数值更新为*this当前值
    // http://en.cppreference.com/w/cpp/atomic/atomic/compare_exchange
  }
  return false;
}

void ClockCacheShard::Unref(CacheHandle* handle, bool set_usage,
                            CleanupContext* context) {
  if (set_usage) {
    handle->flags.fetch_or(kUsageBit, std::memory_order_relaxed);
    //@NOTE fetch_or 原子的逻辑或操作，返回操作前的变量值，即先取值再做或计算
  }
  // Use acquire-release semantics as previous operations on the cache entry
  // has to be order before reference count is decreased, and potential cleanup
  // of the entry has to be order after.
  uint32_t flags = handle->flags.fetch_sub(kOneRef, std::memory_order_acq_rel);
  assert(CountRefs(flags) > 0);
  if (CountRefs(flags) == 1) {
    // this is the last reference.
    pinned_usage_.fetch_sub(handle->charge, std::memory_order_relaxed);
    // Cleanup if it is the last reference.
    if (!InCache(flags)) {
      MutexLock l(&mutex_);
      RecycleHandle(handle, context);
    }
  }
}

void ClockCacheShard::UnsetInCache(CacheHandle* handle,
                                   CleanupContext* context) {
  mutex_.AssertHeld();
  // Use acquire-release semantics as previous operations on the cache entry
  // has to be order before reference count is decreased, and potential cleanup
  // of the entry has to be order after.
  uint32_t flags =
      handle->flags.fetch_and(~kInCacheBit, std::memory_order_acq_rel);
  //@NOTE 先取flag值，再把InCache标记置为0
  // Cleanup if it is the last reference.
  if (InCache(flags) && CountRefs(flags) == 0) {
    RecycleHandle(handle, context);
  }
}

bool ClockCacheShard::TryEvict(CacheHandle* handle, CleanupContext* context) {
  mutex_.AssertHeld();
  uint32_t flags = kInCacheBit;
  //@NOTE kInCacheBit直接作为flags时表示CacheHandle在Cache中、未使用且引用计数为0
  if (handle->flags.compare_exchange_strong(flags, 0, std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
    bool erased __attribute__((__unused__)) =
        table_.erase(CacheKey(handle->key, handle->hash));
    assert(erased);
    RecycleHandle(handle, context);
    return true;
  }
  handle->flags.fetch_and(~kUsageBit, std::memory_order_relaxed);
  return false;
}

bool ClockCacheShard::EvictFromCache(size_t charge, CleanupContext* context) {
  size_t usage = usage_.load(std::memory_order_relaxed);
  size_t capacity = capacity_.load(std::memory_order_relaxed);
  if (usage == 0) {
    return charge <= capacity;
  }
  size_t new_head = head_;
  bool second_iteration = false;
  while (usage + charge > capacity) {
    assert(new_head < list_.size());
    if (TryEvict(&list_[new_head], context)) {
      usage = usage_.load(std::memory_order_relaxed);
    }
    new_head = (new_head + 1 >= list_.size()) ? 0 : new_head + 1;
    if (new_head == head_) {
      if (second_iteration) {
        //@NOTE 扫两轮，如果空间不够返回false
        return false;
      } else {
        second_iteration = true;
      }
    }
  }
  head_ = new_head;
  return true;
}

void ClockCacheShard::SetCapacity(size_t capacity) {
  CleanupContext context;
  {
    MutexLock l(&mutex_);
    capacity_.store(capacity, std::memory_order_relaxed);
    EvictFromCache(0, &context);
  }
  Cleanup(context);
  //@NOTE 更新缓存容量后做一次回收..
}

void ClockCacheShard::SetStrictCapacityLimit(bool strict_capacity_limit) {
  strict_capacity_limit_.store(strict_capacity_limit,
                               std::memory_order_relaxed);
}

CacheHandle* ClockCacheShard::Insert(
    const Slice& key, uint32_t hash, void* value, size_t charge,
    void (*deleter)(const Slice& key, void* value), bool hold_reference,
    CleanupContext* context) {
  MutexLock l(&mutex_);
  bool success = EvictFromCache(charge, context);
  bool strict = strict_capacity_limit_.load(std::memory_order_relaxed);
  if (!success && (strict || !hold_reference)) {
    //@NOTE 没有缓存空间且(严格容量限制 或 参数资源是临时引用) 则释放资源
    //资源-形参key,value指向的内存空间...
    context->to_delete_key.push_back(key.data());
    if (!hold_reference) {
      context->to_delete_value.emplace_back(key, value, deleter);
    }
    //@NOTE hold_reference 表示当前方法的调用者是否保存了资源的引用。
    //如果没有，则回收资源
    return nullptr;
  }
  // Grab available handle from recycle bin. If recycle bin is empty, create
  // and append new handle to end of circular list.
  CacheHandle* handle = nullptr;
  if (!recycle_.empty()) {
    handle = recycle_.back();
    recycle_.pop_back();
  } else {
    list_.emplace_back();
    handle = &list_.back();
  }
  // Fill handle.
  handle->key = key;
  handle->hash = hash;
  handle->value = value;
  handle->charge = charge;
  handle->deleter = deleter;
  uint32_t flags = hold_reference ? kInCacheBit + kOneRef : kInCacheBit;
  handle->flags.store(flags, std::memory_order_relaxed);
  HashTable::accessor accessor;
  if (table_.find(accessor, CacheKey(key, hash))) {
    CacheHandle* existing_handle = accessor->second;
    table_.erase(accessor);
    UnsetInCache(existing_handle, context);
    //@NOTE 把相同key的旧value踢出去
  }
  table_.insert(HashTable::value_type(CacheKey(key, hash), handle));
  if (hold_reference) {
    pinned_usage_.fetch_add(charge, std::memory_order_relaxed);
  }
  usage_.fetch_add(charge, std::memory_order_relaxed);
  return handle;
}

Status ClockCacheShard::Insert(const Slice& key, uint32_t hash, void* value,
                               size_t charge,
                               void (*deleter)(const Slice& key, void* value),
                               Cache::Handle** out_handle,
                               Cache::Priority priority) {
  CleanupContext context;
  HashTable::accessor accessor;
  //@NOTE 无效代码
  char* key_data = new char[key.size()];
  memcpy(key_data, key.data(), key.size());
  Slice key_copy(key_data, key.size());
  CacheHandle* handle = Insert(key_copy, hash, value, charge, deleter,
                               out_handle != nullptr, &context);
  //@NOTE 为什么要为key申请空间？另一个Insert函数并没有这么做
  //这个方法是公开接口，key内存由调用者负责，value内存交给ClockCache负责。

  Status s;
  if (out_handle != nullptr) {
    if (handle == nullptr) {
      s = Status::Incomplete("Insert failed due to LRU cache being full.");
      //@NOTE LRU cache ??? CLOCK cache
    } else {
      *out_handle = reinterpret_cast<Cache::Handle*>(handle);
    }
  }
  Cleanup(context);
  return s;
}

Cache::Handle* ClockCacheShard::Lookup(const Slice& key, uint32_t hash) {
  HashTable::const_accessor accessor;
  if (!table_.find(accessor, CacheKey(key, hash))) {
    return nullptr;
  }
  CacheHandle* handle = accessor->second;
  accessor.release();
  // Ref() could fail if another thread sneak in and evict/erase the cache
  // entry before we are able to hold reference.
  if (!Ref(reinterpret_cast<Cache::Handle*>(handle))) {
    return nullptr;
  }
  // Double check the key since the handle may now representing another key
  // if other threads sneak in, evict/erase the entry and re-used the handle
  // for another cache entry.
  if (hash != handle->hash || key != handle->key) {
    CleanupContext context;
    Unref(handle, false, &context);
    // It is possible Unref() delete the entry, so we need to cleanup.
    Cleanup(context);
    return nullptr;
  }
  return reinterpret_cast<Cache::Handle*>(handle);
}

void ClockCacheShard::Release(Cache::Handle* h) {
  CleanupContext context;
  CacheHandle* handle = reinterpret_cast<CacheHandle*>(h);
  Unref(handle, true, &context);
  Cleanup(context);
}

void ClockCacheShard::Erase(const Slice& key, uint32_t hash) {
  CleanupContext context;
  {
    MutexLock l(&mutex_);
    HashTable::accessor accessor;
    if (table_.find(accessor, CacheKey(key, hash))) {
      CacheHandle* handle = accessor->second;
      table_.erase(accessor);
      UnsetInCache(handle, &context);
    }
  }
  Cleanup(context);
}

void ClockCacheShard::EraseUnRefEntries() {
  CleanupContext context;
  {
    MutexLock l(&mutex_);
    table_.clear();
    for (auto& handle : list_) {
      UnsetInCache(&handle, &context);
    }
  }
  Cleanup(context);
}

class ClockCache : public ShardedCache {
//@NOTE ShardedCache和CacheShard有啥区别？
//ShardedCache - 分片化的缓存，内部有若干个分片，分别存储数据
//CacheShard - 一个缓存分片，实际存储逻辑
 public:
  ClockCache(size_t capacity, int num_shard_bits, bool strict_capacity_limit)
      : ShardedCache(capacity, num_shard_bits, strict_capacity_limit) {
    int num_shards = 1 << num_shard_bits;
    shards_ = new ClockCacheShard[num_shards];
    SetCapacity(capacity);
    SetStrictCapacityLimit(strict_capacity_limit);
  }

  virtual ~ClockCache() { delete[] shards_; }

  virtual const char* Name() const override { return "ClockCache"; }

  virtual CacheShard* GetShard(int shard) override {
    return reinterpret_cast<CacheShard*>(&shards_[shard]);
  }

  virtual const CacheShard* GetShard(int shard) const override {
    return reinterpret_cast<CacheShard*>(&shards_[shard]);
  }

  virtual void* Value(Handle* handle) override {
    return reinterpret_cast<const CacheHandle*>(handle)->value;
  }

  virtual size_t GetCharge(Handle* handle) const override {
    return reinterpret_cast<const CacheHandle*>(handle)->charge;
  }

  virtual uint32_t GetHash(Handle* handle) const override {
    return reinterpret_cast<const CacheHandle*>(handle)->hash;
  }

  virtual void DisownData() override { shards_ = nullptr; }
  //@NOTE 析构函数delete[] null 会挂吗？ 确实不会挂，delete会忽视nullptr...

 private:
  ClockCacheShard* shards_;
};

}  // end anonymous namespace

std::shared_ptr<Cache> NewClockCache(size_t capacity, int num_shard_bits,
                                     bool strict_capacity_limit) {
  if (num_shard_bits < 0) {
    num_shard_bits = GetDefaultCacheShardBits(capacity);
  }
  return std::make_shared<ClockCache>(capacity, num_shard_bits,
                                      strict_capacity_limit);
  //@NOTE std::make_shared 创建对象并返回shared_ptr
  //对比std::shared_ptr，make_shared将创建T对象、构造共享指针合并为一次内存申请，
  //而shared_ptr，需要申请内存创建T对象，然后申请内存创建shared_ptr
  // http://en.cppreference.com/w/cpp/memory/shared_ptr/make_shared
}

}  // namespace rocksdb

#endif  // SUPPORT_CLOCK_CACHE
