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
 * \file compressor.cc
 * \brief Gradient compression for kvstore
 * \author Shuo Ouyang
 */

#include "compressor.h"
#include "../kvstore_local.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::kvstore::compressor::CompressorReg);
}  // namespace dmlc

namespace mxnet {
namespace kvstore {
namespace compressor {
void Compressor::CompressEx(const mxnet::NDArray& from, mxnet::NDArray* to,
                            mxnet::NDArray* residual, const int priority) {
  CHECK(shape_is_known(from.shape())) << "source operand has undefined shape";
  CHECK(shape_is_known(to->shape())) << "destination operand has undefined shape";
  CHECK(shape_is_known(residual->shape())) << "residual operand has undefined shape";

  const int a = from.ctx().dev_mask();
  const int b = to->ctx().dev_mask();
  if (a == mshadow::cpu::kDevMask and b == mshadow::cpu::kDevMask) {
    auto compress_cpu = [this, from, to, residual](mxnet::RunContext rctx) {
      this->Compress(rctx, from.data(), const_cast<mxnet::TBlob&>(to->data()),
                     const_cast<mxnet::TBlob&>(residual->data()));
    };
    mxnet::Engine::Get()->PushSync(compress_cpu, from.ctx(), {from.var()},
                                   {to->var(), residual->var()}, mxnet::FnProperty::kNormal,
                                   priority, "CompressCPU");
  } else {
#if MXNET_USE_CUDA
    if (a == mshadow::gpu::kDevMask and b == mshadow::gpu::kDevMask) {
      auto compress_gpu = [this, from, to, residual](mxnet::RunContext rctx) {
        this->Compress(rctx, from.data(), const_cast<mxnet::TBlob&>(to->data()),
                       const_cast<mxnet::TBlob&>(residual->data()));
        rctx.get_stream<mshadow::gpu>()->Wait();
      };
      mxnet::Engine::Get()->PushSync(compress_gpu, from.ctx(), {from.var()},
                                     {to->var(), residual->var()}, mxnet::FnProperty::kNormal,
                                     priority, "CompressGPU");
    } else {
      LOG(FATAL) << "Unknown device mask.";
    }
#else
    LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
  }
}

void Compressor::DecompressEx(const mxnet::NDArray& from, mxnet::NDArray* to, const int priority) {
  CHECK(shape_is_known(from.shape())) << "source operand has undefined shape";
  CHECK(shape_is_known(to->shape())) << "destination operand has undefined shape";
  const int a = from.ctx().dev_mask();
  const int b = to->ctx().dev_mask();
  if (a == mshadow::cpu::kDevMask and b == mshadow::cpu::kDevMask) {
    auto decompress_cpu = [this, from, to](mxnet::RunContext rctx) {
      this->Decompress(rctx, from.data(), const_cast<mxnet::TBlob&>(to->data()));
    };
    mxnet::Engine::Get()->PushSync(decompress_cpu, from.ctx(), {from.var()}, {to->var()},
                                   mxnet::FnProperty::kNormal, priority, "DecompressCPU");
  } else {
#if MXNET_USE_CUDA
    if (a == mshadow::gpu::kDevMask and b == mshadow::gpu::kDevMask) {
      auto decompress_gpu = [this, from, to, priority](mxnet::RunContext rctx) {
        this->Decompress(rctx, from.data(), const_cast<mxnet::TBlob&>(to->data()));
        rctx.get_stream<mshadow::gpu>()->Wait();
      };
      mxnet::Engine::Get()->PushSync(decompress_gpu, from.ctx(), {from.var()}, {to->var()},
                                     mxnet::FnProperty::kNormal, priority, "DecompressGPU");
    } else {
      LOG(FATAL) << "Unknown device mask.";
    }
#else
    LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
  }
}

kwarg_t Compressor::DecodeParams(const std::string& s) const {
  std::vector<std::string> elems;
  mxnet::kvstore::split(s, ',', std::back_inserter(elems));
  kwarg_t kwargs;
  CHECK_EQ(elems.size() % 2, 0U) << "must have two times elements";
  for (size_t i = 0; i < elems.size(); i += 2) {
    kwargs.emplace_back(elems[i], elems[i + 1]);
  }
  return kwargs;
}

Compressor* Compressor::Create(const char* type_name) {
  auto* creator = dmlc::Registry<CompressorReg>::Find(type_name);
  if (creator == nullptr) {
    LOG(FATAL) << "Cannot find Compressor " << type_name << " in registry";
  }
  return creator->body();
}

}  // namespace compressor
}  // namespace kvstore
}  // namespace mxnet
