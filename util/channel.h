//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under the BSD-style license found in the
//  LICENSE file in the root directory of this source tree. An additional grant
//  of patent rights can be found in the PATENTS file in the same directory.

#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

#pragma once

namespace rocksdb {

template <class T>
class channel {
 public:
  explicit channel() : eof_(false) {}

  channel(const channel&) = delete;
  void operator=(const channel&) = delete;

  void sendEof() {
    std::lock_guard<std::mutex> lk(lock_);
    eof_ = true;
    cv_.notify_all();
  }
  //@NOTE std::lock_guard提供了一种方便的RAII机制，构造时抢锁、析构时释放锁
  // http://en.cppreference.com/w/cpp/thread/lock_guard
  // https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization

  bool eof() {
    std::lock_guard<std::mutex> lk(lock_);
    return buffer_.empty() && eof_;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lk(lock_);
    return buffer_.size();
  }

  // writes elem to the queue
  void write(T&& elem) {
    std::unique_lock<std::mutex> lk(lock_);
    buffer_.emplace(std::forward<T>(elem));
    //@NOTE std::forward 
    //std::forward只有在它的参数绑定到一个右值上的时候，它才转换它的参数到一个右值。
    //std::move执行到右值的无条件转换。就其本身而言，它没有move任何东西。
    //std::move和std::forward在运行期都没有做任何事情。
    // http://www.cnblogs.com/boydfd/p/5182743.html
    // http://en.cppreference.com/w/cpp/utility/forward
    cv_.notify_one();
  }
  //@NOTE 这里unique_lock和lock_guard有区别吗？
  //unique_lock可以替代lock_guard，比lock_guard更灵活同时需要更多的性能。
  //lock_guard只能支持构造时抢锁、析构时释放，
  //如果需要wait,notify这种在生命周期内灵活操作条件等待、通知，unique_lock能够满足要求。
  //http://stackoverflow.com/questions/20516773/stdunique-lockstdmutex-or-stdlock-guardstdmutex

  /// Moves a dequeued element onto elem, blocking until an element
  /// is available.
  // returns false if EOF
  bool read(T& elem) {
    std::unique_lock<std::mutex> lk(lock_);
    cv_.wait(lk, [&] { return eof_ || !buffer_.empty(); });
    if (eof_ && buffer_.empty()) {
      return false;
    }
    elem = std::move(buffer_.front());
    buffer_.pop();
    cv_.notify_one();
    return true;
  }

 private:
  std::condition_variable cv_;
  //@NOTE condition_variable 条件变量，
  // wait: 释放锁，等待直到被唤醒且满足指定条件时重新抢到锁
  // notify_one,notify_all：唤醒关联到统一互斥锁的条件变量
  // http://en.cppreference.com/w/cpp/thread/condition_variable
  std::mutex lock_;
  std::queue<T> buffer_;
  bool eof_;
};
}  // namespace rocksdb
