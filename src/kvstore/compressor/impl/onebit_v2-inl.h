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

#ifndef MXNET_KVSTORE_COMPRESSOR_IMPL_ONEBIT_V2_INL_H_
#define MXNET_KVSTORE_COMPRESSOR_IMPL_ONEBIT_V2_INL_H_

#include "../../../operator/mxnet_op.h"
#include "../../../operator/operator_common.h"
#include "../compressor.h"

namespace mxnet {
namespace kvstore {
namespace compressor {

struct OneBitCompressorV2Param : public dmlc::Parameter<OneBitCompressorV2Param> {
  float threshold;
  float ef_alpha;
  DMLC_DECLARE_PARAMETER(OneBitCompressorV2Param) {
    DMLC_DECLARE_FIELD(threshold).set_default(0).describe(
        "Threshold to use for onebit gradient compression");
    DMLC_DECLARE_FIELD(ef_alpha).set_default(1).describe("Alpha for momentum error feedback");
  }
};

struct QuantizeOneBitV2CPUKernel {
  MSHADOW_XINLINE static void Map(int block_id, int original_size, float *compr_grad, float *grad,
                                  float *residual, const float threshold, const float alpha) {
    float *compr_block = compr_grad + block_id;
    *compr_block = 0;
    const int start = block_id << 5;
    const int end = (start + 32 <= original_size) ? start + 32 : original_size;
    const uint8_t bits[] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
    char *block_ptr = reinterpret_cast<char *>(compr_block);
    for (int i = start; i < end; ++i) {
      residual[i] = (1 - alpha) * residual[i] + alpha * grad[i];
      char *curr_byte = block_ptr + ((i - start) >> 3);
      if (residual[i] > threshold) {
        *curr_byte |= bits[i & 7];
        residual[i] -= 1;
      } else {
        residual[i] += 1;
      }
    }
  }
};

struct DequantizeOneBitV2CPUKernel {
  MSHADOW_XINLINE static void Map(int block_id, int original_size, float *grad, float *compr_grad,
                                  const float threshold) {
    float *compr_block = compr_grad + block_id;
    const int start = block_id << 5;
    const int end = (start + 32 <= original_size) ? start + 32 : original_size;
    const uint8_t bits[] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
    char *block_ptr = reinterpret_cast<char *>(compr_block);
    for (int i = start; i < end; ++i) {
      char *curr_byte = block_ptr + ((i - start) >> 3);
      const uint8_t mask = bits[i & 7];
      const uint8_t masked = *curr_byte & mask;
      if (masked == mask) {
        grad[i] = 1;
      } else {
        grad[i] = -1;
      }
    }
  }
};

template <typename xpu>
void AccumulateGradientCompute(mshadow::Stream<xpu> *s, const mxnet::TBlob &in, mxnet::TBlob *out,
                               const float alpha) {
  mxnet::op::mxnet_op::Kernel<AccumulateGradientKernel, xpu>::Launch(s, in.Size(), in.dptr<float>(),
                                                                     out->dptr<float>(), alpha);
}

template <typename xpu>
void UpdateErrorCompute(mshadow::Stream<xpu> *s, mxnet::TBlob *out, const float threshold) {
  mxnet::op::mxnet_op::Kernel<UpdateErrorKernel, xpu>::Launch(s, out->Size(), out->dptr<float>(),
                                                              threshold);
}

template <typename xpu>
void QuantizeOneBitV2Compute(mshadow::Stream<xpu> *s, const mxnet::TBlob &in, mxnet::TBlob *out,
                             mxnet::TBlob *residual, const float threshold, const float alpha) {
  mxnet::op::mxnet_op::Kernel<QuantizeOneBitV2CPUKernel, xpu>::Launch(
      s,
      out->Size(),              // compressed array size
      in.Size(),                // original array size
      out->dptr<float>(),       // compressed array
      in.dptr<float>(),         // original array
      residual->dptr<float>(),  // residual array
      threshold,                // threshold
      alpha                     // alpha for error feedback
  );
}

template <typename xpu>
void DequantizeOneBitV2Compute(mshadow::Stream<xpu> *s, const mxnet::TBlob &in, mxnet::TBlob *out,
                               const float threshold) {
  mxnet::op::mxnet_op::Kernel<DequantizeOneBitV2CPUKernel, xpu>::Launch(
      s,
      in.Size(),           // compressed array size
      out->Size(),         // original size
      out->dptr<float>(),  // out array
      in.dptr<float>(),    // compressed array
      threshold            // threshold
  );
}

#if MXNET_USE_CUDA
void AccumulateGradientCompute(mshadow::Stream<mshadow::gpu> *s, const mxnet::TBlob &in,
                               mxnet::TBlob *out, const float alpha);

void UpdateErrorCompute(mshadow::Stream<mshadow::gpu> *s, mxnet::TBlob *out, const float threshold);

void QuantizeOneBitV2Compute(mshadow::Stream<mshadow::gpu> *s, const mxnet::TBlob &in,
                             mxnet::TBlob *out, const float threshold);

void DequantizeOneBitV2Compute(mshadow::Stream<mshadow::gpu> *s, const mxnet::TBlob &in,
                               mxnet::TBlob *out, const float threshold);
#endif

class OneBitCompressorV2 : public Compressor {
 public:
  explicit OneBitCompressorV2() = default;

  ~OneBitCompressorV2() = default;

  void Init(const kwarg_t &kwargs) override { param_.InitAllowUnknown(kwargs); }

  inline int GetCompressFactor() const override { return 32; }

  inline bool SupportFastAggregate() const override { return false; }

  std::map<std::string, std::string> GetParams() const override { return param_.__DICT__(); }

  std::string TypeString() const override { return "OneBitCompressorV2"; }

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

 private:
  template <typename xpu>
  void CompressImpl(mxnet::RunContext &rctx, const mxnet::TBlob &in, mxnet::TBlob *out,
                    mxnet::TBlob *residual) {
    AccumulateGradientCompute(rctx.get_stream<xpu>(), in, residual, param_.ef_alpha);
    QuantizeOneBitV2Compute(rctx.get_stream<xpu>(), (*residual), out, param_.threshold);
    UpdateErrorCompute(rctx.get_stream<xpu>(), residual, param_.threshold);
  }

  template <typename xpu>
  void DecompressImpl(mxnet::RunContext &rctx, const mxnet::TBlob &in, mxnet::TBlob *out) {
    DequantizeOneBitV2Compute(rctx.get_stream<xpu>(), in, out, param_.threshold);
  }

  OneBitCompressorV2Param param_;
};

}  // namespace compressor
}  // namespace kvstore
}  // namespace mxnet

#endif
