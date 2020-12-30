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

#ifndef MXNET_KVSTORE_COMPRESSOR_IMPL_TWOBIT_V2_INL_H_
#define MXNET_KVSTORE_COMPRESSOR_IMPL_TWOBIT_V2_INL_H_

#include "../compressor.h"
#include "../../../operator/mxnet_op.h"
#include "../../../operator/operator_common.h"

namespace mxnet {
namespace kvstore {
namespace compressor {

struct TwoBitCompressorV2Param : public dmlc::Parameter<TwoBitCompressorV2Param> {
  float threshold;
  float ef_alpha;
  DMLC_DECLARE_PARAMETER(TwoBitCompressorV2Param) {
    DMLC_DECLARE_FIELD(threshold).set_default(0.5).describe(
        "Threshold to use for 2bit gradient compression");
    DMLC_DECLARE_FIELD(ef_alpha).set_default(1).describe("Alpha for momentum error feedback");
  }
};

struct QuantizeTwoBitV2CPUKernel {
  MSHADOW_XINLINE static void Map(int block_id, int original_size, float *compr_grad, float *grad,
                                  float *residual, const int threshold, const float alpha) {
    float *compr_block = compr_grad + block_id;
    *compr_block = 0;

    const int start = block_id << 4;
    const int end = (start + 16 <= original_size) ? start + 16 : original_size;

    char *block_ptr = reinterpret_cast<char *>(compr_block);

    const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};

    for (int i = start; i < end; ++i) {
      residual[i] = (1 - alpha) * residual[i] + alpha * grad[i];
      char *curr_byte = block_ptr + ((i - start) >> 2);
      if (residual[i] >= threshold) {
        *curr_byte |= posbits[i & 3];
        residual[i] -= threshold;
      } else if (residual[i] <= threshold) {
        *curr_byte |= negbits[i & 3];
        residual[i] += threshold;
      }
    }
  }
};

struct DequantizeTwoBitV2CPUKernel {
  MSHADOW_XINLINE static void Map(int block_id, int original_size, float *grad, float *compr_grad,
                                  const float threshold) {
    float *compr_block = compr_grad + block_id;
    const int start = block_id << 4;
    const int end = (start + 16 <= original_size) ? start + 16 : original_size;

    char *block_ptr = reinterpret_cast<char *>(compr_block);

    const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};

    for (int i = start; i < end; ++i) {
      char *curr_byte = block_ptr + ((i - start) >> 2);
      const uint8_t posmask = posbits[i & 3];
      const uint8_t negmask = negbits[i & 3];
      const uint8_t masked = *curr_byte & posmask;
      if (masked == posmask) {
        grad[i] = threshold;
      } else if (masked == negmask) {
        grad[i] = -threshold;
      } else {
        grad[i] = 0;
      }
    }
  }
};

template <typename xpu>
void QuantizeTwoBitV2Compute(mshadow::Stream<xpu> *s, const mxnet::TBlob &in, mxnet::TBlob *out,
                             mxnet::TBlob *residual, const float threshold, const float alpha) {
  mxnet::op::mxnet_op::Kernel<QuantizeTwoBitV2CPUKernel, xpu>::Launch(
      s,
      out->Size(),              // compressed array size
      in.Size(),                // original size
      out->dptr<float>(),       // compressed array
      in.dptr<float>(),         // original array
      residual->dptr<float>(),  // residual array
      threshold,                // threshold
      alpha);
}

template <typename xpu>
void DequantizeTwoBitV2Compute(mshadow::Stream<xpu> *s, const mxnet::TBlob &in, mxnet::TBlob *out,
                               const float threshold) {
  mxnet::op::mxnet_op::Kernel<DequantizeTwoBitV2CPUKernel, xpu>::Launch(
      s,
      in.Size(),           // compressed array size
      out->Size(),         // original size
      out->dptr<float>(),  // out array
      in.dptr<float>(),    // compressed array
      threshold);          // threshold
}

#if MXNET_USE_CUDA
void QuantizeTwoBitV2Compute(mshadow::Stream<mshadow::gpu> *s, const mxnet::TBlob &in,
                             mxnet::TBlob *out, mxnet::TBlob *residual, float threshold,
                             const float alpha);

void DequantizeTwoBitV2Compute(mshadow::Stream<mshadow::gpu> *s, const mxnet::TBlob &in,
                               mxnet::TBlob *out, const float threshold);
#endif

class TwoBitCompressorV2 : public Compressor {
 public:
  explicit TwoBitCompressorV2() = default;

  ~TwoBitCompressorV2() = default;

  void Init(const kwarg_t &kwargs) override {
    param_.InitAllowUnknown(kwargs);
    CHECK_GT(param_.threshold, 0) << "threshod for two bit quantization must large than 0.";
  }

  std::string TypeString() const override { return "TwoBitCompressorV2"; }

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
    QuantizeTwoBitV2Compute(rctx.get_stream<xpu>(), in, out, residual, param_.threshold,
                            param_.ef_alpha);
  }

  template <typename xpu>
  void DecompressImpl(mxnet::RunContext &rctx, const mxnet::TBlob &in, mxnet::TBlob *out) {
    DequantizeTwoBitV2Compute(rctx.get_stream<xpu>(), in, out, param_.threshold);
  }

  TwoBitCompressorV2Param param_;
};

}  // namespace compressor
}  // namespace kvstore
}  // namespace mxnet

#endif