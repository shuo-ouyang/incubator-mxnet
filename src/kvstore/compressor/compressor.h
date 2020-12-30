/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file compressor.h
 * \brief Gradient compression for kvstore
 * \author Shuo Ouyang
 */

#ifndef MXNET_KVSTORE_COMPRESSOR_COMPRESSOR_H_
#define MXNET_KVSTORE_COMPRESSOR_COMPRESSOR_H_

#include <functional>
#include <dmlc/parameter.h>
#include <dmlc/registry.h>
#include <string>
#include <utility>
#include <vector>
#include "mxnet/ndarray.h"

namespace mxnet {
namespace kvstore {
namespace compressor {

using kwarg_t = std::vector<std::pair<std::string, std::string>>;

class Compressor {
 public:
  Compressor() = default;

  virtual ~Compressor() = default;

  virtual void Init(const kwarg_t &) = 0;

  virtual int GetCompressFactor() const = 0;

  virtual int64_t GetCompressedSize(const int64_t &original_size) {
    const int factor = this->GetCompressFactor();
    return ((original_size % factor == 0) ? original_size / factor : original_size / factor + 1);
  }

  virtual bool SupportFastAggregate() const = 0;

  virtual std::map<std::string, std::string> GetParams() const = 0;

  static Compressor *Create(const char *);

  virtual std::string TypeString() const = 0;

  virtual void Compress(mxnet::RunContext &, const mxnet::TBlob &, mxnet::TBlob *,
                        mxnet::TBlob *) = 0;

  virtual void Decompress(mxnet::RunContext &, const mxnet::TBlob &, mxnet::TBlob *) = 0;

  virtual void DecompressAndAggregate(mxnet::RunContext &, const mxnet::TBlob &,
                                      mxnet::TBlob *) = 0;
};

using CompressorFactory = std::function<Compressor *()>;

struct CompressorReg : public dmlc::FunctionRegEntryBase<CompressorReg, CompressorFactory> {
  inline CompressorReg &check_name() {
    Compressor *compr = this->body();
    std::string type = compr->TypeString();
    delete compr;
    CHECK_EQ(this->name, type) << "Register Name and TypeString mismatch, name=\"" << this->name
                               << "\","
                               << " but TypeString=\"" << type << "\"";
    return *this;
  }
};

#if MXNET_USE_CUDA
#define METHOD_DISPATCH(Method, ...)     \
  if (ctx.dev_mask() == cpu::kDevMask) { \
    Method<cpu>(__VA_ARGS__);            \
  } else {                               \
    Method<gpu>(__VA_ARGS__);            \
  }
#else
#define METHOD_DISPATCH(Method, ...)     \
  if (ctx.dev_mask() == cpu::kDevMask) { \
    Method<cpu>(__VA_ARGS__);            \
  } else {                               \
    LOG(FATAL) << "GPU is not enabled";  \
  }
#endif

#define KVSTORE_REGISTER_COMPRESSOR(name, CompressorType)                                  \
  DMLC_REGISTRY_REGISTER(::mxnet::kvstore::compressor::CompressorReg, CompressorReg, name) \
      .set_body([]() { return new CompressorType(); })                                     \
      .set_return_type("NDArray-or-Symbol")                                                \
      .check_name()

}  // namespace compressor
}  // namespace kvstore
}  // namespace mxnet

#endif