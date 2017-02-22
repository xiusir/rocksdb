//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under the BSD-style license found in the
//  LICENSE file in the root directory of this source tree. An additional grant
//  of patent rights can be found in the PATENTS file in the same directory.
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
#pragma once

#include <algorithm>
#include "port/port.h"

namespace rocksdb {

//@NOTE 将s指向一个内存page的边界, page_size必须是2的幂
// 所以page_size-1的结果二进制表示肯定是若干个1的组合
// s & (page_size - 1) 即取s的二进制末尾x位，x = log2(page_size)
// s = s - (s & (page_size - 1)) 相当于将s向0的方向移动到当前page的起始位置
inline size_t TruncateToPageBoundary(size_t page_size, size_t s) {
  s -= (s & (page_size - 1));
  assert((s % page_size) == 0);
  return s;
}

//@NOTE 将指针x后移到下一个内存块(标准size)的起始位置，y即内存块的size
inline size_t Roundup(size_t x, size_t y) {
  return ((x + y - 1) / y) * y;
}

// This class is to manage an aligned user
// allocated buffer for direct I/O purposes
// though can be used for any purpose.
class AlignedBuffer {
  size_t alignment_;
  std::unique_ptr<char[]> buf_;
  //@NOTE buf_ 类型相当于char[]* 的智能指针
  size_t capacity_;
  size_t cursize_;
  char* bufstart_;

  //@NOTE std::unique_ptr智能指针，保证正常和异常情况下指针被适当处理

public:
  AlignedBuffer()
    : alignment_(),
      capacity_(0),
      cursize_(0),
      bufstart_(nullptr) {
  }

  AlignedBuffer(AlignedBuffer&& o) ROCKSDB_NOEXCEPT {
    //@NOTE 推荐《Effective Modern C++》，上面讲得很清楚
    //C++11中，T&&这种语法形式有两种意思：
    //  右值引用（Rvalue reference），只能绑定右值
    //  万能引用（Universal reference），既能绑定右值，又能绑定左值

    //@NOTE ROCKSDB_NOEXCEPT 即noexcept 表示函数不应抛出异常
    //如果抛出了异常，会直接调用std::terminate终止运行
    //即有异常就直接退出，不需要调用者进行异常处理
    *this = std::move(o);
    //@NOTE std::move和std::forward仅仅是进行类型转换的函数（实际上是函数模板）
    //std::move无条件的将其参数转换为右值，而std::forward只在必要情况下进行这个转换
    //move后，变量o失效，其关联的对象转移到*this，只能通过*this访问原o引用的对象实例
    //好复杂的解释... http://www.cnblogs.com/chezxiaoqiang/archive/2012/10/24/2736630.html
  }

  AlignedBuffer& operator=(AlignedBuffer&& o) ROCKSDB_NOEXCEPT {
    alignment_ = std::move(o.alignment_);
    buf_ = std::move(o.buf_);
    capacity_ = std::move(o.capacity_);
    cursize_ = std::move(o.cursize_);
    bufstart_ = std::move(o.bufstart_);
    return *this;
  }

  AlignedBuffer(const AlignedBuffer&) = delete;

  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  //@NOTE delete  C11新特性，相当于private修饰，但编译信息更友好
  //用来删任何你不爽的东西，比如拷贝构造，赋值拷贝
  //http://blog.csdn.net/a1875566250/article/details/40406883

  //@NOTE 作为形参，T&& 与const T&有什么区别?
  //lvalue represents an object that occupies some identifiable location in memory
  //rvalue an expression that is not a lvalue...
  //http://eli.thegreenplace.net/2011/12/15/understanding-lvalues-and-rvalues-in-c-and-c
  //C11以前，左值可以被修改，而右值只能被访问不能被修改
  //C11颠覆了这一点，允许我们拿到右值的引用从而修改他们
  //
  //The && syntax is the new rvalue reference. 
  //It does exactly what it sounds it does - gives us a reference to an rvalue, 
  //which is going to be destroyed after the call. 
  //We can use this fact to just "steal" the internals of the rvalue 
  //
  //T&& 是右值引用，其内容注定是要被释放的，所以利用这一点可以将*this引用的空间与
  //右值引用的空间交换，即把现成的东西放到*this标识符下面，把原来*this的东西由右值释放掉
  //std::swap可以用于引用"交换"
  //std::move即将参数转换为右值，将引用传递给赋值号左侧变量，右值成为一个无效引用被释放掉
  //
  //作者使用= delete修饰构造函数、赋值符号重载函数，提高代码性能、不希望Buffer对象被多重引用

  static bool isAligned(const void* ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
  }
  //@NOTE 判断指针ptr是否对齐，将void*转换为uint*后可以进行算术运算
  //typedef unsigned long uintptr_t;

  static bool isAligned(size_t n, size_t alignment) {
    return n % alignment == 0;
  }

  size_t Alignment() const {
    return alignment_;
  }

  size_t Capacity() const {
    return capacity_;
  }

  size_t CurrentSize() const {
    return cursize_;
  }

  const char* BufferStart() const {
    return bufstart_;
  }

  char* BufferStart() { return bufstart_; }

  void Clear() {
    cursize_ = 0;
  }

  void Alignment(size_t alignment) {
    assert(alignment > 0);
    assert((alignment & (alignment - 1)) == 0);
    //@NOTE 判断alignment是否是2的整数次幂，即log2(alignment)是否为整数
    alignment_ = alignment;
  }

  // Allocates a new buffer and sets bufstart_ to the aligned first byte
  void AllocateNewBuffer(size_t requestedCapacity) {

    assert(alignment_ > 0);
    assert((alignment_ & (alignment_ - 1)) == 0);

    size_t size = Roundup(requestedCapacity, alignment_);
    buf_.reset(new char[size + alignment_]);
    //@NOTE 将参数指定的capacity对齐到符合alignment_的位置
    //若capacity满足对齐要求则不变，否则右移至第一个alignment_的整数倍的位置

    char* p = buf_.get();
    bufstart_ = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(p)+(alignment_ - 1)) &
      ~static_cast<uintptr_t>(alignment_ - 1));
    //@NOTE 从buf_的内存区域，取第一个对齐位置指针作为bufstart_
    //原理与Roundup函数相似，将指针右移y-1个位置后做二进制取整，结果即是对齐位置指针
    capacity_ = size;
    cursize_ = 0;
  }
  // Used for write
  // Returns the number of bytes appended
  size_t Append(const char* src, size_t append_size) {
    size_t buffer_remaining = capacity_ - cursize_;
    size_t to_copy = std::min(append_size, buffer_remaining);

    if (to_copy > 0) {
      memcpy(bufstart_ + cursize_, src, to_copy);
      cursize_ += to_copy;
    }
    return to_copy;
  }

  size_t Read(char* dest, size_t offset, size_t read_size) const {
    assert(offset < cursize_);
    size_t to_read = std::min(cursize_ - offset, read_size);
    if (to_read > 0) {
      memcpy(dest, bufstart_ + offset, to_read);
    }
    return to_read;
  }

  /// Pad to alignment
  void PadToAlignmentWith(int padding) {
    size_t total_size = Roundup(cursize_, alignment_);
    size_t pad_size = total_size - cursize_;

    if (pad_size > 0) {
      assert((pad_size + cursize_) <= capacity_);
      memset(bufstart_ + cursize_, padding, pad_size);
      cursize_ += pad_size;
    }
  }
  //@NOTE 向右对齐缓冲区指针，使用(unsigned char)padding填充空洞

  // After a partial flush move the tail to the beginning of the buffer
  void RefitTail(size_t tail_offset, size_t tail_size) {
    if (tail_size > 0) {
      memmove(bufstart_, bufstart_ + tail_offset, tail_size);
    }
    cursize_ = tail_size;
  }

  // Returns place to start writing
  char* Destination() {
    return bufstart_ + cursize_;
  }

  void Size(size_t cursize) {
    cursize_ = cursize;
  }
};
}
