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
