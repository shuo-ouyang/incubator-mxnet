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

#ifndef MXNET_KVSTORE_COMPRESSOR_ERROR_FEEDBACK_INL_H_
#define MXNET_KVSTORE_COMPRESSOR_ERROR_FEEDBACK_INL_H_

#include "compressor.h"
#include "../../operator/mxnet_op.h"

namespace mxnet {
namespace kvstore {
namespace compressor {

struct AccumulateGradientsKernel {
  MSHADOW_XINLINE static void Map(int i, const float* grad, float* residual, float alpha) {
    residual[i] = (1 - alpha) * residual[i] + alpha * grad[i];
  }
};

template <typename xpu>
void AccumulateGradientsKernalLaunch(mshadow::Stream<xpu>* s, const mxnet::TBlob& in,
                                       mxnet::TBlob& out, const float alpha) {
  mxnet::op::mxnet_op::Kernel<AccumulateGradientsKernel, xpu>::Launch(
      s,
      in.Size(),          // array size
      in.dptr<float>(),   // gradient array
      out.dptr<float>(),  // residual array
      alpha);             // momentum
}

inline void AccumulateGradientsImpl(mshadow::Stream<mshadow::cpu>* s, const mxnet::TBlob& in,
                                      mxnet::TBlob& out, const float alpha) {
  AccumulateGradientsKernalLaunch(s, in, out, alpha);
}

#if MXNET_USE_CUDA
void AccumulateGradientsImpl(mshadow::Stream<mshadow::gpu>* s, const mxnet::TBlob& in,
                               mxnet::TBlob& out, const float alpha);

#endif

struct UpdateErrorKernel{
  MSHADOW_XINLINE static void Map(int i, float* residual, const float threshold) {
    
  }
}

}  // namespace compressor
}  // namespace kvstore
}  // namespace mxnet
#endif
