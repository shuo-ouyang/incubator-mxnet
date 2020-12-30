#include "onebit-inl.h"

namespace mxnet {
namespace kvstore {
namespace compressor {
DMLC_REGISTER_PARAMETER(OneBitCompressorParam);

KVSTORE_REGISTER_COMPRESSOR(OneBitCompressor, OneBitCompressor)
    .add_arguments(OneBitCompressorParam::__FIELDS__());

}  // namespace compressor
}  // namespace kvstore
}  // namespace mxnet
