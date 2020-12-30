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
 * \file gradient_compression.cc
 * \brief Gradient compression for kvstore
 * \author Rahul Huilgol
 */

#include <vector>
#include "kvstore_local.h"
#include "gradient_compression.h"

namespace mxnet {
namespace kvstore {

GradientCompression::~GradientCompression() { compr_.release(); }

void GradientCompression::Init(const std::string &name,
                               const mxnet::kvstore::compressor::kwarg_t &kwargs) {
  if (this->IsInitialized()) {
    LOG(WARNING) << "The compressor has been initialized with name " << compr_->TypeString();
    return;
  }
  compr_.reset(mxnet::kvstore::compressor::Compressor::Create(name.c_str()));
  compr_->Init(kwargs);
  this->init_ = true;
}

std::string GradientCompression::get_type_str() { return compr_->TypeString(); }

std::string GradientCompression::EncodeParams() {
  std::string rval = get_type_str();
  auto params = compr_->GetParams();
  for (const auto &kv : params) {
    rval.push_back(',');
    rval += kv.first;
    rval.push_back(',');
    rval += kv.second;
  }
  return rval;
}

std::pair<std::string, kvstore::compressor::kwarg_t> GradientCompression::DecodeParams(
    const std::string &s) {
  std::vector<std::string> elems;
  mxnet::kvstore::split(s, ',', std::back_inserter(elems));
  std::string name = elems[0];
  mxnet::kvstore::compressor::kwarg_t params;
  for (size_t i = 1; i < elems.size(); i += 2) {
    params.emplace_back(elems[i], elems[i + 1]);
  }
  return {name, params};
}

int GradientCompression::GetCompressionFactor() { return compr_->GetCompressFactor(); }

int64_t GradientCompression::GetCompressedSize(const int64_t &original_size) {
  return compr_->GetCompressedSize(original_size);
}

void GradientCompression::CompressEx(const mxnet::NDArray &from, mxnet::NDArray *to,
                                     mxnet::NDArray *residual, const int priority) {
  CHECK(shape_is_known(from.shape())) << "source operand has undefined shape";
  CHECK(shape_is_known(to->shape())) << "destination operand has undefined shape";
  CHECK(shape_is_known(residual->shape())) << "residual operand has undefined shape";
  const int a = from.ctx().dev_mask();
  const int b = to->ctx().dev_mask();

  if (a == mshadow::cpu::kDevMask and b == mshadow::cpu::kDevMask) {
    mxnet::Engine::Get()->PushSync(
        [this, from, to, residual](mxnet::RunContext rctx) {
          compr_->Compress(rctx, from.data(), const_cast<mxnet::TBlob &>(to->data()),
                           const_cast<mxnet::TBlob &>(residual->data()));
        },
        from.ctx(), {from.var()}, {to->var(), residual->var()}, mxnet::FnProperty::kNormal,
        priority, "CompressCPU");
  } else {
#if MXNET_USE_CUDA
    if (a == mshadow::gpu::kDevMask and b == mshadow::gpu::kDevMask) {
      mxnet::Engine::Get()->PushSync(
          [this, from, to, residual](mxnet::RunContext rctx) {
            compr_->Compress(rctx, from.data(), const_cast<mxnet::TBlob &>(to->data()),
                             const_cast<mxnet::TBlob &>(residual->data()));
            // Wait GPU kernel to complete
            rctx.get_stream<mshadow::gpu>()->Wait();
          },
          from.ctx(), {from.var()}, {to->var(), residual->var()}, mxnet::FnProperty::kNormal,
          priority, "CompressGPU");
    } else {
      LOG(FATAL) << "unknown device mask";
    }
#else
    LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
  }
}

void GradientCompression::DecompressEx(const mxnet::NDArray &from, mxnet::NDArray *to,
                                       const int priority) {
  CHECK(shape_is_known(from.shape())) << "source operand has undefined shape";
  CHECK(shape_is_known(to->shape())) << "destination operand has undefined shape";
  const int a = from.ctx().dev_mask();
  const int b = to->ctx().dev_mask();
  if (a == mshadow::cpu::kDevMask and b == mshadow::cpu::kDevMask) {
    mxnet::Engine::Get()->PushSync(
        [this, from, to](mxnet::RunContext rctx) {
          compr_->Decompress(rctx, from.data(), const_cast<mxnet::TBlob &>(to->data()));
        },
        from.ctx(), {from.var()}, {to->var()}, mxnet::FnProperty::kNormal, priority,
        "DecompressCPU");
  } else {
#if MXNET_USE_CUDA
    if (a == mshadow::gpu::kDevMask and b == mshadow::gpu::kDevMask) {
      mxnet::Engine::Get()->PushSync(
          [this, from, to](mxnet::RunContext rctx) {
            std::vector<mxnet::TBlob> inputs = {from.data(), to->data()};
            compr_->Decompress(rctx, from.data(), const_cast<mxnet::TBlob &>(to->data()));
            // Wait GPU kernel to complete
            rctx.get_stream<mshadow::gpu>()->Wait();
          },
          from.ctx(), {from.var()}, {to->var()}, mxnet::FnProperty::kNormal, priority,
          "DecompressGPU");
    } else {
      LOG(FATAL) << "unknown device mask";
    }
#else
    LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
  }
}

}  // namespace kvstore
}  // namespace mxnet
