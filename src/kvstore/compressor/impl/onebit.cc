#include "onebit-inl.h"

namespace mxnet {
namespace kvstore {
namespace compressor {
DMLC_REGISTER_PARAMETER(OneBitCompressorParam);

KVSTORE_REGISTER_COMPRESSOR(OneBitCompressor, OneBitCompressor)
    .add_arguments(OneBitCompressorParam::__FIELDS__());

std::string OneBitCompressor::EncodeParams() const {
  auto param_dict = param_.__DICT__();
  std::string encode_param;
  for (const auto &kv : param_dict) {
    encode_param += kv.first;
    encode_param.push_back(',');
    encode_param += kv.second;
    encode_param.push_back(',');
  }
  encode_param.pop_back();
  return encode_param;
}
}  // namespace compressor
}  // namespace kvstore
}  // namespace mxnet