#include "matrix/kaldi-matrix.h"

namespace {

template<typename Real>
void assert_matrix_shape(const torch::Tensor &tensor_);

template<>
void assert_matrix_shape<float>(const torch::Tensor &tensor_) {
  TORCH_INTERNAL_ASSERT(tensor_.ndimension() == 2);
  TORCH_INTERNAL_ASSERT(tensor_.dtype() == torch::kFloat32);
}

template<>
void assert_matrix_shape<double>(const torch::Tensor &tensor_) {
  TORCH_INTERNAL_ASSERT(tensor_.ndimension() == 2);
  TORCH_INTERNAL_ASSERT(tensor_.dtype() == torch::kFloat64);
}

} // namespace

namespace kaldi {

template<typename Real>
MatrixBase<Real>::MatrixBase(torch::Tensor tensor) : tensor_(tensor) {
  assert_matrix_shape<Real>(tensor_);
};

template struct Matrix<float>;
template struct Matrix<double>;
template struct MatrixBase<float>;
template struct MatrixBase<double>;
template struct SubMatrix<float>;
template struct SubMatrix<double>;

} // namespace kaldi
