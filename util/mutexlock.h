//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under the BSD-style license found in the
//  LICENSE file in the root directory of this source tree. An additional grant
//  of patent rights can be found in the PATENTS file in the same directory.
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#pragma once
#include <assert.h>
#include <atomic>
#include <mutex>
#include <thread>
#include "port/port.h"

namespace rocksdb {

// Helper class that locks a mutex on construction and unlocks the mutex when
// the destructor of the MutexLock object is invoked.
//
// Typical usage:
//
//   void MyClass::MyMethod() {
//     MutexLock l(&mu_);       // mu_ is an instance variable
//     ... some complex code, possibly with multiple return paths ...
//   }

class MutexLock {
 public:
  explicit MutexLock(port::Mutex *mu) : mu_(mu) {
    this->mu_->Lock();
  }
  ~MutexLock() { this->mu_->Unlock(); }

 private:
  port::Mutex *const mu_;
  // No copying allowed
  MutexLock(const MutexLock&);
  void operator=(const MutexLock&);
};

//
// Acquire a ReadLock on the specified RWMutex.
// The Lock will be automatically released then the
// object goes out of scope.
//
class ReadLock {
 public:
  explicit ReadLock(port::RWMutex *mu) : mu_(mu) {
    this->mu_->ReadLock();
  }
  ~ReadLock() { this->mu_->ReadUnlock(); }

 private:
  port::RWMutex *const mu_;
  // No copying allowed
  ReadLock(const ReadLock&);
  void operator=(const ReadLock&);
};

//
// Automatically unlock a locked mutex when the object is destroyed
//
class ReadUnlock {
 public:
  explicit ReadUnlock(port::RWMutex *mu) : mu_(mu) { mu->AssertHeld(); }
  ~ReadUnlock() { mu_->ReadUnlock(); }

 private:
  port::RWMutex *const mu_;
  // No copying allowed
  ReadUnlock(const ReadUnlock &) = delete;
  ReadUnlock &operator=(const ReadUnlock &) = delete;
};

//
// Acquire a WriteLock on the specified RWMutex.
// The Lock will be automatically released then the
// object goes out of scope.
//
class WriteLock {
 public:
  explicit WriteLock(port::RWMutex *mu) : mu_(mu) {
    this->mu_->WriteLock();
  }
  ~WriteLock() { this->mu_->WriteUnlock(); }

 private:
  port::RWMutex *const mu_;
  // No copying allowed
  WriteLock(const WriteLock&);
  void operator=(const WriteLock&);
};

//
// SpinMutex has very low overhead for low-contention cases.  Method names
// are chosen so you can use std::unique_lock or std::lock_guard with it.
//
class SpinMutex {
 public:
  SpinMutex() : locked_(false) {}

  bool try_lock() {
    auto currently_locked = locked_.load(std::memory_order_relaxed);
    return !currently_locked &&
           locked_.compare_exchange_weak(currently_locked, true,
                                         std::memory_order_acquire,
                                         std::memory_order_relaxed);
    //@NOTE 原子cas：若状态为unlocked则置为locked。
    //std::atomic::compare_exchange_weak 返回true表示成功修改变量值。
    //为什么不直接用 compare_exchange_weak(false, true, ...) ?
    //std::atomic::load 代价比compare_exchange_weak低??
    //先进行一次状态判断，可以减少直接compare_exchange_weak = false的代价？
    //
    // http://en.cppreference.com/w/cpp/atomic/atomic/compare_exchange
    //
    // compare_exchange_weak 与 compare_exchange_strong 区别：
    // _weak 允许将compare结果误判为false导致不进行exchange，
    //       所以需要循环执行多次得到可靠结果，
    //       如果原子类型是包含padding bit/trap bit 或相同值对应
    //       多种不同二进制状态(浮点型)，weak效果更好，能快速收敛
    //       到稳定的结果。
    // _strong 强一致性，结果是可靠的，但性能稍差。
  }

  void lock() {
    for (size_t tries = 0;; ++tries) {
      if (try_lock()) {
        // success
        break;
      }
      port::AsmVolatilePause();
      //@NOTE asm pause指令 是专门为自旋锁提供的一个指令，
      //告知处理器当前代码是spin-wait-loop，处理器会根据这个提示而避开
      //内存序列冲突(memory order violation)，也就是说对spin-wait-loop
      //不做缓存，不做指令重新排序等动作。这样就可以大大的提高了处理器的性能。
      //PAUSE另一个功能是让处理器执行spin-wait-loop时减少电源的消耗。
      //在等待资源而执行自旋锁等待时，处理器以极快的速度执行自旋等待，
      //PAUSE 指令实际上就相当于 NOP 指令。
      if (tries > 100) {
        std::this_thread::yield();
        //@NOTE 提示内核当前线程想让出执行资源，休息一会儿，可以避免死锁
        //多个线程抢锁时非常像死循环，一直占用cpu资源，会导致拥有锁的线程
        //无法被及时调度运行释放锁，这样会造成等锁时间非常长，看上去像死锁
        // http://en.cppreference.com/w/cpp/thread/yield
      }
    }
  }

  void unlock() { locked_.store(false, std::memory_order_release); }

 private:
  std::atomic<bool> locked_;
};

}  // namespace rocksdb
