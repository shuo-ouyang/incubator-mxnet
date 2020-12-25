#include "onebit-inl.h"

namespace mxnet {
namespace kvstore {
namespace compressor {
void QuantizeOneBitImpl(mshadow::Stream<mshadow::gpu> *s,
                        const mxnet::TBlob &in, mxnet::TBlob &out,
                        const float threshold) {
  QuantizeOneBitKernelLaunch(s, in, out, threshold);
}

void DequantizeOneBitImpl(mshadow::Stream<mshadow::gpu> *s,
                          const mxnet::TBlob &in, mxnet::TBlob &out,
                          const float threshold) {
  DequantizeOneBitKernelLaunch(s, in, out, threshold);
}
} // namespace compressor
} // namespace kvstore
} // namespace mxnet