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
 * \file compressor.cc
 * \brief Gradient compression for kvstore
 * \author Shuo Ouyang
 */

#include "compressor.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::kvstore::compressor::CompressorReg);
}  // namespace dmlc

namespace mxnet {
namespace kvstore {
namespace compressor {

Compressor* Compressor::Create(const char* type_name) {
  auto* creator = dmlc::Registry<CompressorReg>::Find(type_name);
  if (creator == nullptr) {
    LOG(FATAL) << "Cannot find Compressor " << type_name << " in registry";
  }
  return creator->body();
}

}  // namespace compressor
}  // namespace kvstore
}  // namespace mxnet