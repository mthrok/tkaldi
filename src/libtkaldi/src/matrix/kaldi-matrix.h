// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h

#ifndef KALDI_MATRIX_KALDI_MATRIX_H_
#define KALDI_MATRIX_KALDI_MATRIX_H_

#include <torch/torch.h>
#include <matrix/matrix-common.h>

using namespace torch::indexing;

namespace kaldi {

namespace {

template<typename Real>
void assert_matrix_shape(const torch::Tensor &tensor_) {
  TORCH_INTERNAL_ASSERT(tensor_.ndimension() == 2);
  static_assert(std::is_same<Real, float>::value, "Matrix class only supports float32.");
  TORCH_INTERNAL_ASSERT(tensor_.dtype() == torch::kFloat32);
}

} // namespace

template<typename Real> struct VectorBase;

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L44-L48
template<typename Real>
struct MatrixBase {
  ////////////////////////////////////////////////////////////////////////////////
  // PyTorch-specific items
  ////////////////////////////////////////////////////////////////////////////////
  torch::Tensor tensor_;
  /// Construct VectorBase which is an interface to an existing torch::Tensor object.
  MatrixBase(torch::Tensor tensor) : tensor_(tensor) {
    assert_matrix_shape<Real>(tensor_);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Kaldi-compatible items
  ////////////////////////////////////////////////////////////////////////////////
  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L62-L63
  inline MatrixIndexT NumRows() const { return tensor_.size(0); };

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L65-L66
  inline MatrixIndexT NumCols() const { return tensor_.size(1); };

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L177-L178
  void CopyColFromVec(const VectorBase<Real> &v, const MatrixIndexT col) {
    tensor_.index_put_({Slice(), Slice(col)}, v.tensor_);
  }
};

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L940-L948
template<typename Real>
struct SubMatrix : MatrixBase<Real> {
  SubMatrix(const MatrixBase<Real>& T,
            const MatrixIndexT ro,  // row offset, 0 < ro < NumRows()
            const MatrixIndexT r,   // number of rows, r > 0
            const MatrixIndexT co,  // column offset, 0 < co < NumCols()
            const MatrixIndexT c)   // number of columns, c > 0
    : MatrixBase<Real>(T.tensor_.index({Slice(ro, ro+r), Slice(co, co+c)})) {}
};
 
  
} // namespace kaldi

#endif
