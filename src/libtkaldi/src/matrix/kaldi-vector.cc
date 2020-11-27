// matrix/kaldi-vector.cc

// Copyright 2009-2011  Microsoft Corporation;  Lukas Burget;
//                      Saarland University;   Go Vivace Inc.;  Ariya Rastrow;
//                      Petr Schwarz;  Yanmin Qian;  Jan Silovsky;
//                      Haihua Xu; Wei Shi
//                2015  Guoguo Chen
//                2017  Daniel Galvez


// See https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/COPYING
// for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"

namespace {

template<typename Real>
void assert_vector_shape(const torch::Tensor &tensor_);

template<>
void assert_vector_shape<float>(const torch::Tensor &tensor_) {
  TORCH_INTERNAL_ASSERT(tensor_.ndimension() == 1);
  TORCH_INTERNAL_ASSERT(tensor_.dtype() == torch::kFloat32);
}

template<>
void assert_vector_shape<double>(const torch::Tensor &tensor_) {
  TORCH_INTERNAL_ASSERT(tensor_.ndimension() == 1);
  TORCH_INTERNAL_ASSERT(tensor_.dtype() == torch::kFloat64);
}

} // namespace

namespace kaldi {

template<typename Real>
VectorBase<Real>::VectorBase(torch::Tensor tensor) : tensor_(tensor) {
  assert_vector_shape<Real>(tensor_);
};

template<typename Real>
VectorBase<Real>::VectorBase() : tensor_(torch::empty({0})) {
  assert_vector_shape<Real>(tensor_);
}

template struct Vector<float>;
template struct Vector<double>;
template struct VectorBase<float>;
template struct VectorBase<double>;

} // namespace kaldi
