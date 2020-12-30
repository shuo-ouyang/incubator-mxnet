#include "onebit_v2-inl.h"

namespace mxnet {
namespace kvstore {
namespace compressor {

void QuantizeOneBitV2Compute(mshadow::Stream<mshadow::gpu> *s,
                             const mxnet::TBlob &in, mxnet::TBlob *out,
                             mxnet::TBlob *residual, const float threshold,
                             const float alpha) {
  mxnet::op::mxnet_op::Kernel<QuantizeOneBitV2Kernel, mshadow::gpu>::Launch(
      s,
      in.Size(),         // original array size
      out->dptr<float>(), // compressed array
      in.dptr<float>(),  // original array
      residual->dptr<float>(),
      threshold,          // threshold
      alpha
  );
}

void DequantizeOneBitV2Compute(mshadow::Stream<mshadow::gpu> *s,
                               const mxnet::TBlob &in, mxnet::TBlob *out,
                               const float threshold) {
  mxnet::op::mxnet_op::Kernel<DequantizeOneBitV2Kernel, mshadow::gpu>::Launch(
      s,
      out->Size(),        // original size
      out->dptr<float>(), // out array
      in.dptr<float>(),  // compressed array
      threshold          // threshold
  );
}

void DequantizeAndAggregateOneBitV2Compute(mshadow::Stream<mshadow::gpu> *s, const mxnet::TBlob &in,
                                           mxnet::TBlob *out) {
  mxnet::op::mxnet_op::Kernel<DequantizeAndAggregateOneBitV2Kernel, mshadow::gpu>::Launch(
      s,
      out->Size(),         // original size
      out->dptr<float>(),  // gradient array
      in.dptr<float>()    // compressed array
  );
}

} // namespace compressor
} // namespace kvstore
} // namespace mxnet
