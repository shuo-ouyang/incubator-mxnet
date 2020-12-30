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

#include "twobit_v2-inl.h"

namespace mxnet {
namespace kvstore {
namespace compressor {

struct QuantizeTwoBitV2GPUKernel {
  MSHADOW_XINLINE static void Map(int byte_id, int original_size,
                                  float *compr_grad, float *grad,
                                  float *residual, const float threshold,
                                  const float alpha) {
    const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};

    char *compr_byte = reinterpret_cast<char *>(compr_grad) + byte_id;
    *compr_byte = 0;

    const int start = byte_id << 2;
    const int end = (start + 4 <= original_size) ? start + 4 : original_size;

    for (int i = start; i < end; ++i) {
      residual[i] = (1 - alpha) * residual[i] + alpha * grad[i];
      if (residual[i] >= threshold) {
        *compr_byte |= posbits[i & 3];
        residual[i] -= threshold;
      } else if (residual[i] <= -threshold) {
        *compr_byte |= negbits[i & 3];
        residual[i] += threshold;
      }
    }
  }
};

struct DequantizeTwoBitV2GPUKernel {
  MSHADOW_XINLINE static void Map(int i, float *out, float *in,
                                  const float threshold) {
    char *ch_ptr = reinterpret_cast<char *>(in + (i >> 4));
    ch_ptr += ((i & 15) >> 2);

    const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};

    const uint8_t posmask = posbits[i & 3];
    const uint8_t negmask = negbits[i & 3];
    const uint8_t masked = *ch_ptr & posmask;
    if (masked == posmask) {
      out[i] = threshold;
    } else if (masked == negmask) {
      out[i] = -threshold;
    } else {
      out[i] = 0;
    }
  }
};

void QuantizeTwoBitV2Compute(mshadow::Stream<mshadow::gpu> *s,
                             const mxnet::TBlob &in, mxnet::TBlob *out,
                             mxnet::TBlob *residual, const float threshold,
                             const float alpha) {
  mxnet::op::mxnet_op::Kernel<QuantizeTwoBitV2GPUKernel, mshadow::gpu>::Launch(
      s,
      out->Size() * 4,         // compressed array size
      in.Size(),               // original size
      out->dptr<float>(),      // compressed array
      in.dptr<float>(),        // original array
      residual->dptr<float>(), // residual array
      threshold,               // threshold
      alpha);
}

void DequantizeTwoBitV2Compute(mshadow::Stream<mshadow::gpu> *s,
                               const mxnet::TBlob &in, mxnet::TBlob *out,
                               const float threshold) {
  mxnet::op::mxnet_op::Kernel<DequantizeTwoBitV2GPUKernel, mshadow::gpu>::
      Launch(s,
             out->Size(),        // original size
             out->dptr<float>(), // out array
             in.dptr<float>(),   // compressed array
             threshold);         // threshold
}
} // namespace compressor
} // namespace kvstore
} // namespace mxnet
