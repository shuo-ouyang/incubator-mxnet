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
 * \file twobit.h
 * \brief Two bit compressor for kvstore.
 * \author Shuo Ouyang
 */

#ifndef MXNET_KVSTORE_COMPRESSOR_IMPL_TWOBIT_INL_H_
#define MXNET_KVSTORE_COMPRESSOR_IMPL_TWOBIT_INL_H_

#include "../compressor.h"
#include "../../../operator/mxnet_op.h"
#include "../../../operator/operator_common.h"

namespace mxnet {
namespace kvstore {
namespace compressor {

struct TwoBitCompressorParam : public dmlc::Parameter<TwoBitCompressorParam> {
  float threshold;
  float ef_alpha;
  DMLC_DECLARE_PARAMETER(TwoBitCompressorParam) {
    DMLC_DECLARE_FIELD(threshold).set_default(0.5).describe(
        "Threshold to use for 2bit gradient compression");
    DMLC_DECLARE_FIELD(ef_alpha).set_default(1).describe("Alpha for momentum error feedback");
  }
};

struct QuantizeTwoBitKernel {
  MSHADOW_XINLINE static void Map(int out_block_id, int original_size, float *compr_grad,
                                  float *grad, float *residual, const float neg_threshold,
                                  const float pos_threshold, const float alpha) {
    // this block contains the compressed representation of
    // upto 16 values starting from out_block_id*16
    float *compr_block = compr_grad + out_block_id;
    // init to 0
    *compr_block = 0;
    // start and end are indices in original grad array
    const int start = out_block_id << 4;
    const int end = (start + 16 <= original_size) ? start + 16 : original_size;
    // cast as char* to manipulate bits of float addresses
    char *block_ptr = reinterpret_cast<char *>(compr_block);
    // masks to set bits when value meets pos_threshold
    // 0xc0 is mask when value is to be represented by the first two bits in a
    // char* 0xc0 means first two bits are set to 11
    const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    // masks to set bits when value meets neg_threshold
    const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};
    for (int i = start; i < end; i++) {
      // adds offset to reach appropriate byte
      char *curr_byte = block_ptr + ((i - start) >> 2);
      // adds gradient to existing residual to get updated grad
      residual[i] = (1 - alpha) * residual[i] + alpha * grad[i];
      if (residual[i] >= pos_threshold) {
        // set data to 11
        *curr_byte |= posbits[(i & 3)];
        residual[i] -= pos_threshold;
      } else if (residual[i] <= neg_threshold) {
        // set data to 10
        *curr_byte |= negbits[(i & 3)];
        residual[i] -= neg_threshold;
      }
    }
  }
};

struct DequantizeTwoBitKernel {
  MSHADOW_XINLINE static void Map(int i, float *out, float *in, const float neg_threshold,
                                  const float pos_threshold) {
    // get position of dequantized value to fill
    float *outval = out + i;
    // gets byte which holds quantized value for this position
    char *ch_ptr = reinterpret_cast<char *>(in + (i >> 4));
    ch_ptr += ((i & 15) >> 2);
    // masks used to quantize data
    const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};
    // col denotes which two bits of a byte are set for this value
    // col=0 implies first two bits, col=3 implies last two bits,...
    const int col = i & 3;
    const uint8_t mask = posbits[col];
    const uint8_t negmask = negbits[col];
    const uint8_t masked = *ch_ptr & mask;
    if (masked == mask) {
      *outval = pos_threshold;
    } else if (masked == negmask) {
      // use posbits for mask as posbits are both 1s
      // then compare masked with negbits to see if only negbits were set
      *outval = neg_threshold;
    } else {
      *outval = 0;
    }
  }
};

template <typename xpu>
void QuantizeTwoBitCompute(mshadow::Stream<xpu> *s, const mxnet::TBlob &in, mxnet::TBlob *out,
                           mxnet::TBlob *residual, const float threshold, const float alpha) {
  mxnet::op::mxnet_op::Kernel<QuantizeTwoBitKernel, xpu>::Launch(
      s,
      out->Size(),         // compressed array size
      in.Size(),           // original size
      out->dptr<float>(),  // compressed array
      in.dptr<float>(),    // original array
      residual->dptr<float>(),
      -threshold,  // negative threshold
      threshold,   // positive threshold
      alpha);
}

template <typename xpu>
void DequantizeTwoBitCompute(mshadow::Stream<xpu> *s, const mxnet::TBlob &in, mxnet::TBlob *out,
                             const float threshold) {
  mxnet::op::mxnet_op::Kernel<DequantizeTwoBitKernel, xpu>::Launch(
      s,
      out->Size(),         // original size
      out->dptr<float>(),  // out array
      in.dptr<float>(),   // compressed array
      -threshold,         // negative threshold
      threshold);         // positive threshold
}

#if MXNET_USE_CUDA
void QuantizeTwoBitCompute(mshadow::Stream<mshadow::gpu> *s, const mxnet::TBlob &in,
                           mxnet::TBlob *out, mxnet::TBlob *residual, float threshold,
                           const float alpha);

void DequantizeTwoBitCompute(mshadow::Stream<mshadow::gpu> *s, const mxnet::TBlob &in,
                             mxnet::TBlob *out, const float threshold);
#endif

class TwoBitCompressor : public Compressor {
 public:
  explicit TwoBitCompressor() = default;

  ~TwoBitCompressor() = default;

  void Init(const kwarg_t &kwargs) override {
    param_.InitAllowUnknown(kwargs);
    CHECK_GT(param_.threshold, 0) << "threshod for two bit quantization must large than 0.";
  }

  std::string TypeString() const override { return "TwoBitCompressor"; }

  inline bool SupportFastAggregate() const override { return false; }

  std::map<std::string, std::string> GetParams() const override { return param_.__DICT__(); }

 protected:
  virtual void Compress(mxnet::RunContext &rctx, const mxnet::TBlob &in, mxnet::TBlob *out,
                        mxnet::TBlob *residual) override {
    mxnet::Context ctx = rctx.get_ctx();
    METHOD_DISPATCH(CompressImpl, rctx, in, out, residual);
  }

  virtual void Decompress(mxnet::RunContext &rctx, const mxnet::TBlob &in,
                          mxnet::TBlob *out) override {
    mxnet::Context ctx = rctx.get_ctx();
    METHOD_DISPATCH(DecompressImpl, rctx, in, out);
  }

  virtual void DecompressAndAggregate(mxnet::RunContext &rctx, const mxnet::TBlob &in,
                                      mxnet::TBlob *out) override {
    LOG(FATAL) << "Not Implemented!";
  }

  inline int GetCompressFactor() const override { return 16; }

 private:
  template <typename xpu>
  void CompressImpl(mxnet::RunContext &rctx, const mxnet::TBlob &in, mxnet::TBlob *out,
                    mxnet::TBlob *residual) {
    QuantizeTwoBitCompute(rctx.get_stream<xpu>(), in, out, residual, param_.threshold,
                          param_.ef_alpha);
  }

  template <typename xpu>
  void DecompressImpl(mxnet::RunContext &rctx, const mxnet::TBlob &in, mxnet::TBlob *out) {
    DequantizeTwoBitCompute(rctx.get_stream<xpu>(), in, out, param_.threshold);
  }

  TwoBitCompressorParam param_;
};

}  // namespace compressor
}  // namespace kvstore
}  // namespace mxnet

#endif