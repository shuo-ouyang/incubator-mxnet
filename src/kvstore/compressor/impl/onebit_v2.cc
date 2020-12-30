#include "onebit_v2-inl.h"

namespace mxnet {
namespace kvstore {
namespace compressor {
DMLC_REGISTER_PARAMETER(OneBitCompressorV2Param);

KVSTORE_REGISTER_COMPRESSOR(OneBitCompressorV2, OneBitCompressorV2)
    .add_arguments(OneBitCompressorV2Param::__FIELDS__());

}  // namespace compressor
}  // namespace kvstore
}  // namespace mxnet
