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
 * \file twobit.cu
 * \author Shuo Ouyang
 * \brief Implementation for gpu version of twobit compression
 */

#include "twobit-inl.h"

namespace mxnet {
namespace kvstore {
namespace compressor {

void QuantizeTwoBitCompute(mshadow::Stream<mshadow::gpu> *s,
                           const mxnet::TBlob &in, mxnet::TBlob *out,
                           mxnet::TBlob *residual, const float threshold,
                           const float alpha) {
  mxnet::op::mxnet_op::Kernel<QuantizeTwoBitKernel, mshadow::gpu>::Launch(
      s,
      out->Size() * 4,         // compressed array size
      in.Size(),               // original size
      out->dptr<float>(),      // compressed array
      in.dptr<float>(),        // original array
      residual->dptr<float>(), // residual array
      threshold,               // threshold
      alpha);
}

void DequantizeTwoBitCompute(mshadow::Stream<mshadow::gpu> *s,
                             const mxnet::TBlob &in, mxnet::TBlob *out,
                             const float threshold) {
  mxnet::op::mxnet_op::Kernel<DequantizeTwoBitKernel, mshadow::gpu>::Launch(
      s,
      out->Size(),        // original size
      out->dptr<float>(), // out array
      in.dptr<float>(),   // compressed array
      threshold);         // threshold
}
} // namespace compressor
} // namespace kvstore
} // namespace mxnet
