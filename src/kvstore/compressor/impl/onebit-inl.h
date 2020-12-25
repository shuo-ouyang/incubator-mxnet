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

#include <mxnet/runtime/packed_func.h>
#include <mxnet/runtime/registry.h>
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
  MSHADOW_XINLINE static void Map(int out_block_id, int original_size, float *out, float *grad,
                                  const float threshold) {
    // this block contains the compressed representation of
    // upto 32 values starting from out_block_id*32
    float *compr_block = out + out_block_id;
    // init to 0
    *compr_block = 0;
    // start and end are indices in original grad array
    const int start = out_block_id << 5;
    const int end = (start + 32 <= original_size) ? start + 32 : original_size;

    char *block_ptr = reinterpret_cast<char *>(compr_block);
    // masks used to quantize data
    const uint8_t bits[] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
    for (int i = start; i < end; ++i) {
      // adds offset to reach appropriate byte
      char *curr_byte = block_ptr + ((i - start) >> 3);
      // adds gradient to existing residual to get updated grad
      if (grad[i] > threshold) {
        // set data to 1
        *curr_byte |= bits[(i & 7)];
      }
    }
  }
};

struct DequantizeOneBitKernel {
  MSHADOW_XINLINE static void Map(int i, float *out, float *in, const float threshold) {
    // get position of dequantized value to fill
    float *outval = out + i;
    // gets byte which holds quantized value for this position
    char *ch_ptr = reinterpret_cast<char *>(in + (i >> 5));
    ch_ptr += ((i & 31) >> 3);
    // masks used to quantize data
    const uint8_t bits[] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
    // col denotes which bit of a byte is set for this value
    // col=0 implies the first bit, col=1 implies the second bit,...
    const int col = i & 7;
    const uint8_t mask = bits[col];
    const uint8_t masked = *ch_ptr & mask;
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
void QuantizeOneBitKernelLaunch(mshadow::Stream<xpu> *s, const mxnet::TBlob &in, mxnet::TBlob &out,
                                const float threshold) {
  mxnet::op::mxnet_op::Kernel<QuantizeOneBitKernel, xpu>::Launch(
      s,
      out.Size(),         // compressed array size
      in.Size(),          // original array size
      out.dptr<float>(),  // compressed array
      in.dptr<float>(),   // original array
      threshold           // threshold
  );
}

template <typename xpu>
void DequantizeOneBitKernelLaunch(mshadow::Stream<xpu> *s, const mxnet::TBlob &in,
                                  mxnet::TBlob &out, const float threshold) {
  mxnet::op::mxnet_op::Kernel<DequantizeOneBitKernel, xpu>::Launch(
      s,
      out.Size(),         // original size
      out.dptr<float>(),  // out array
      in.dptr<float>(),   // compressed array
      threshold           // threshold
  );
}

inline void QuantizeOneBitImpl(mshadow::Stream<mshadow::cpu> *s, const mxnet::TBlob &in,
                               mxnet::TBlob &out, const float threshold) {
  QuantizeOneBitKernelLaunch(s, in, out, threshold);
}

inline void DequantizeOneBitImpl(mshadow::Stream<mshadow::cpu> *s, const mxnet::TBlob &in,
                                 mxnet::TBlob &out, const float threshold) {
  DequantizeOneBitKernelLaunch(s, in, out, threshold);
}

#if MXNET_USE_CUDA
void QuantizeOneBitImpl(mshadow::Stream<mshadow::gpu> *s, const mxnet::TBlob &in, mxnet::TBlob &out,
                        const float threshold);

void DequantizeOneBitImpl(mshadow::Stream<mshadow::gpu> *s, const mxnet::TBlob &in,
                          mxnet::TBlob &out, const float threshold);
#endif

class OneBitCompressor : public Compressor {
 public:
  explicit OneBitCompressor() = default;

  ~OneBitCompressor() = default;

  void Init(const kwarg_t &kwargs) override {
    if (this->IsInitialized()) {
      LOG(WARNING) << "Try to double init!";
      return;
    }
    param_.InitAllowUnknown(kwargs);
    this->init_ = true;
  }

  std::string EncodeParams() const override;

  inline int GetCompressFactor() const override { return 32; }

  std::string TypeString() const override { return "OneBitCompressor"; }

 protected:
  virtual void Compress(mxnet::RunContext &rctx, const mxnet::TBlob &in, mxnet::TBlob &out,
                        mxnet::TBlob &residual) override {
    mxnet::Context ctx = rctx.get_ctx();
    BIND_DISPATCH(CompressImpl, rctx, in, out, residual);
  }

  virtual void Decompress(mxnet::RunContext &rctx, const mxnet::TBlob &in,
                          mxnet::TBlob &out) override {
    mxnet::Context ctx = rctx.get_ctx();
    BIND_DISPATCH(DecompressImpl, rctx, in, out);
  }

 private:
  template <typename xpu>
  void CompressImpl(mxnet::RunContext &rctx, const mxnet::TBlob &in, mxnet::TBlob &out,
                    mxnet::TBlob &residual) {
    MomentumErrorFeedbackImpl(rctx.get_stream<xpu>(), in, residual, param_.ef_alpha);
    QuantizeOneBitImpl(rctx.get_stream<xpu>(), residual, out, param_.threshold);
  }

  template <typename xpu>
  void DecompressImpl(mxnet::RunContext &rctx, const mxnet::TBlob &in, mxnet::TBlob &out) {
    DequantizeOneBitImpl(rctx.get_stream<xpu>(), in, out, param_.threshold);
  }

  OneBitCompressorParam param_;
};

}  // namespace compressor
}  // namespace kvstore
}  // namespace mxnet

#endif