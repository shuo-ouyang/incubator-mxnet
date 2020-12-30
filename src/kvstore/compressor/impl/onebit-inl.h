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
 * \file onebit-inl.h
 * \brief One bit compressor for kvstore.
 * \author Shuo Ouyang
 */

#ifndef MXNET_KVSTORE_COMPRESSOR_IMPL_ONEBIT_H_
#define MXNET_KVSTORE_COMPRESSOR_IMPL_ONEBIT_H_

#include "../compressor.h"
#include "../error_feedback-inl.h"
#include "../../../operator/mxnet_op.h"
#include "../../../operator/operator_common.h"

namespace mxnet {
namespace kvstore {
namespace compressor {

struct OneBitCompressorParam : public dmlc::Parameter<OneBitCompressorParam> {
  float threshold;
  float ef_alpha;
  DMLC_DECLARE_PARAMETER(OneBitCompressorParam) {
    DMLC_DECLARE_FIELD(threshold).set_default(0).describe(
        "Threshold to use for onebit gradient compression");
    DMLC_DECLARE_FIELD(ef_alpha).set_default(1).describe("Alpha for momentum error feedback");
  }
};

struct QuantizeOneBitKernel {
  MSHADOW_XINLINE static void Map(int i, float *out, float *in, const float threshold) {
    float *inval = in + i;
    char *compr_block = reinterpret_cast<char *>(out + (i >> 5));
    char *curr_byte = compr_block + ((i & 0x1f) >> 3);
    const uint8_t mask = 1 << (7 - (i & 7));
    if (*inval >= threshold) {
      *curr_byte |= mask;
    } else {
      *curr_byte &= ~mask;
    }
  }
};

struct DequantizeOneBitKernel {
  MSHADOW_XINLINE static void Map(int i, float *out, float *in, const float threshold) {
    // get position of dequantized value to fill
    float *outval = out + i;
    // gets byte which holds quantized value for this position
    char *compr_blcok = reinterpret_cast<char *>(in + (i >> 5));
    char *curr_byte = compr_blcok + ((i & 0x1f) >> 3);
    // masks used to quantize data
    const uint8_t mask = 1 << (7 - (i & 7));
    const uint8_t masked = *curr_byte & mask;
    if (masked == mask) {
      *outval = +1;
    } else {
      // if current position of byte is 0
      // dequantized it to -1
      *outval = -1;
    }
  }
};

template <typename xpu>
void QuantizeOneBitCompute(mshadow::Stream<xpu> *s, const mxnet::TBlob &in, mxnet::TBlob &out,
                           const float threshold) {
  mxnet::op::mxnet_op::Kernel<QuantizeOneBitKernel, xpu>::Launch(
      s,
      in.Size(),          // original array size
      out.dptr<float>(),  // compressed array
      in.dptr<float>(),   // original array
      threshold           // threshold
  );
}

template <typename xpu>
void DequantizeOneBitCompute(mshadow::Stream<xpu> *s, const mxnet::TBlob &in, mxnet::TBlob &out,
                             const float threshold) {
  mxnet::op::mxnet_op::Kernel<DequantizeOneBitKernel, xpu>::Launch(
      s,
      out.Size(),         // original size
      out.dptr<float>(),  // out array
      in.dptr<float>(),   // compressed array
      threshold           // threshold
  );
}
#if MXNET_USE_CUDA
void QuantizeOneBitCompute(mshadow::Stream<mshadow::gpu> *s, const mxnet::TBlob &in,
                           mxnet::TBlob &out, const float threshold);

void DequantizeOneBitCompute(mshadow::Stream<mshadow::gpu> *s, const mxnet::TBlob &in,
                             mxnet::TBlob &out, const float threshold);
#endif

class OneBitCompressor : public Compressor {
 public:
  explicit OneBitCompressor() = default;

  ~OneBitCompressor() = default;

  void Init(const kwarg_t &kwargs) override { param_.InitAllowUnknown(kwargs); }

  inline int GetCompressFactor() const override { return 32; }

  std::map<std::string, std::string> GetParams() const override { return param_.__DICT__(); }

  std::string TypeString() const override { return "OneBitCompressor"; }

  virtual void Compress(mxnet::RunContext &rctx, const mxnet::TBlob &in, mxnet::TBlob &out,
                        mxnet::TBlob &residual) override {
    mxnet::Context ctx = rctx.get_ctx();
    METHOD_DISPATCH(CompressImpl, rctx, in, out, residual);
  }

  virtual void Decompress(mxnet::RunContext &rctx, const mxnet::TBlob &in,
                          mxnet::TBlob &out) override {
    mxnet::Context ctx = rctx.get_ctx();
    METHOD_DISPATCH(DecompressImpl, rctx, in, out);
  }

 private:
  template <typename xpu>
  void CompressImpl(mxnet::RunContext &rctx, const mxnet::TBlob &in, mxnet::TBlob &out,
                    mxnet::TBlob &residual) {
    MomentumErrorFeedbackImpl(rctx.get_stream<xpu>(), in, residual, param_.ef_alpha);
    QuantizeOneBitCompute(rctx.get_stream<xpu>(), residual, out, param_.threshold);
  }

  template <typename xpu>
  void DecompressImpl(mxnet::RunContext &rctx, const mxnet::TBlob &in, mxnet::TBlob &out) {
    DequantizeOneBitCompute(rctx.get_stream<xpu>(), in, out, param_.threshold);
  }

  OneBitCompressorParam param_;
};

}  // namespace compressor
}  // namespace kvstore
}  // namespace mxnet

#endif
