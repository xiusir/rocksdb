//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under the BSD-style license found in the
//  LICENSE file in the root directory of this source tree. An additional grant
//  of patent rights can be found in the PATENTS file in the same directory.
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "util/coding.h"

#include <algorithm>
#include "rocksdb/slice.h"
#include "rocksdb/slice_transform.h"

namespace rocksdb {

char* EncodeVarint32(char* dst, uint32_t v) {
//@NOTE 编码方法：将v的32个bit按7个1组分成5组，从高位bit所在分组开始将所有2进制
//值为0的分组丢弃，直到遇到第一个不为0的分组停止。
//然后按从低位分组开始将7bit依次按顺序写入dst的每个byte的低7bit。
//将每个byte的最高位bit作为标志位，0表示当前字节为最后一个有效byte不需要继续读取，
//1表示当前byte的下一个byte也是有效byte。
//即使用最高位为0标识32位整型编码的结束byte。
// big-endian
//所以v最多会占用5个byte，最少占用1个byte，v越大占用的字节可能越多。
//
//适用场景：大多数v值小于2^21的场景。
  // Operate on characters as unsigneds
  unsigned char* ptr = reinterpret_cast<unsigned char*>(dst);
  static const int B = 128;
  if (v < (1 << 7)) {
    *(ptr++) = v;
  } else if (v < (1 << 14)) {
    *(ptr++) = v | B;
    *(ptr++) = v >> 7;
  } else if (v < (1 << 21)) {
    *(ptr++) = v | B;
    *(ptr++) = (v >> 7) | B;
    *(ptr++) = v >> 14;
  } else if (v < (1 << 28)) {
    *(ptr++) = v | B;
    *(ptr++) = (v >> 7) | B;
    *(ptr++) = (v >> 14) | B;
    *(ptr++) = v >> 21;
  } else {
    *(ptr++) = v | B;
    *(ptr++) = (v >> 7) | B;
    *(ptr++) = (v >> 14) | B;
    *(ptr++) = (v >> 21) | B;
    *(ptr++) = v >> 28;
  }
  return reinterpret_cast<char*>(ptr);
  //@NOTE 这套C++代码里几乎没有强制类型转换，
  //使用reinterpret_cast<T>更安全。
}

const char* GetVarint32PtrFallback(const char* p, const char* limit,
                                   uint32_t* value) {
//@NOTE 按EncodeVarint32编码方式从p读取value，返回下一段数据的头指针。
  uint32_t result = 0;
  for (uint32_t shift = 0; shift <= 28 && p < limit; shift += 7) {
    uint32_t byte = *(reinterpret_cast<const unsigned char*>(p));
    p++;
    if (byte & 128) {
      // More bytes are present
      result |= ((byte & 127) << shift);
    } else {
      result |= (byte << shift);
      *value = result;
      return reinterpret_cast<const char*>(p);
    }
  }
  return nullptr;
}

const char* GetVarint64Ptr(const char* p, const char* limit, uint64_t* value) {
//@NOTE 和GetVarint32PtrFallback逻辑完全相同。
  uint64_t result = 0;
  for (uint32_t shift = 0; shift <= 63 && p < limit; shift += 7) {
    uint64_t byte = *(reinterpret_cast<const unsigned char*>(p));
    p++;
    if (byte & 128) {
      // More bytes are present
      result |= ((byte & 127) << shift);
    } else {
      result |= (byte << shift);
      *value = result;
      return reinterpret_cast<const char*>(p);
    }
  }
  return nullptr;
}

}  // namespace rocksdb
