#include "onebit-inl.h"

namespace mxnet {
namespace kvstore {
namespace compressor {

void QuantizeOneBitCompute(mshadow::Stream<mshadow::gpu> *s,
                           const mxnet::TBlob &in, mxnet::TBlob &out,
                           const float threshold) {
  mxnet::op::mxnet_op::Kernel<QuantizeOneBitKernel, mshadow::gpu>::Launch(
      s,
      in.Size(),         // original array size
      out.dptr<float>(), // compressed array
      in.dptr<float>(),  // original array
      threshold          // threshold
  );
}

void DequantizeOneBitCompute(mshadow::Stream<mshadow::gpu> *s,
                             const mxnet::TBlob &in, mxnet::TBlob &out,
                             const float threshold) {
  mxnet::op::mxnet_op::Kernel<DequantizeOneBitKernel, mshadow::gpu>::Launch(
      s,
      out.Size(),        // original size
      out.dptr<float>(), // out array
      in.dptr<float>(),  // compressed array
      threshold          // threshold
  );
}
} // namespace compressor
} // namespace kvstore
} // namespace mxnet
