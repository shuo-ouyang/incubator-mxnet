#include "onebit_v2-inl.h"

namespace mxnet {
namespace kvstore {
namespace compressor {

struct QuantizeOneBitV2GPUKernel {
  MSHADOW_XINLINE static void Map(int byte_id, int original_size,
                                  float *compr_grad, float *grad,
                                  float *residual, const float threshold,
                                  const float alpha) {
    const int start = byte_id << 3;
    const int end = (start + 8 <= original_size) ? start + 8 : original_size;
    const uint8_t bits[] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
    char *compr_byte = reinterpret_cast<char *>(compr_grad) + byte_id;
    *compr_byte = 0;
    for (int i = start; i < end; ++i) {
      residual[i] = (1 - alpha) * residual[i] + alpha * grad[i];
      if (residual[i] > threshold) {
        *compr_byte |= bits[i & 7];
        residual[i] -= 1;
      } else {
        residual[i] += 1;
      }
    }
  }
};

struct DequantizeOneBitV2GPUKernel {
  MSHADOW_XINLINE static void Map(int i, float *grad, float *compr_grad,
                                  const float threshold) {

    char *compr_blcok = reinterpret_cast<char *>(compr_grad + (i >> 5));
    char *curr_byte = compr_blcok + ((i & 0x1f) >> 3);
    const uint8_t bits[] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
    const uint8_t mask = bits[i & 7];
    const uint8_t masked = *curr_byte & mask;
    if (masked == mask) {
      grad[i] = 1;
    } else {
      grad[i] = -1;
    }
  }
};

void QuantizeOneBitV2Compute(mshadow::Stream<mshadow::gpu> *s,
                             const mxnet::TBlob &in, mxnet::TBlob *out,
                             mxnet::TBlob *residual, const float threshold,
                             const float alpha) {
  mxnet::op::mxnet_op::Kernel<QuantizeOneBitV2GPUKernel, mshadow::gpu>::Launch(
      s,
      in.Size(),               // original array size
      out->Size() * 4,         // compressed array byte size
      out->dptr<float>(),      // compressed array
      in.dptr<float>(),        // original array
      residual->dptr<float>(), // residual array
      threshold,               // threshold
      alpha);
}

void DequantizeOneBitV2Compute(mshadow::Stream<mshadow::gpu> *s,
                               const mxnet::TBlob &in, mxnet::TBlob *out,
                               const float threshold) {
  mxnet::op::mxnet_op::Kernel<DequantizeOneBitV2GPUKernel, mshadow::gpu>::Launch(
      s,
      out->Size(),        // original size
      out->dptr<float>(), // out array
      in.dptr<float>(),   // compressed array
      threshold           // threshold
  );
}
} // namespace compressor
} // namespace kvstore
} // namespace mxnet
