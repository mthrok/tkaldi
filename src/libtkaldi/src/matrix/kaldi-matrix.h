// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h

#ifndef KALDI_MATRIX_KALDI_MATRIX_H_
#define KALDI_MATRIX_KALDI_MATRIX_H_

#include <torch/script.h>
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
  /// Returns number of rows (or zero for emtpy matrix).
  inline MatrixIndexT NumRows() const { return tensor_.size(0); };

  /// Returns number of columns (or zero for emtpy matrix).
  inline MatrixIndexT NumCols() const { return tensor_.size(1); };

  /// Copy vector into specific column of matrix.
  void CopyColFromVec(const VectorBase<Real> &v, const MatrixIndexT col) {
    tensor_.index_put_({Slice(), Slice(col)}, v.tensor_);
  }
};

/**
  Sub-matrix representation.
  Can work with sub-parts of a matrix using this class.
  Note that SubMatrix is not very const-correct-- it allows you to
  change the contents of a const Matrix.  Be careful!
*/
template<typename Real>
struct SubMatrix : MatrixBase<Real> {
  // Initialize a SubMatrix from part of a matrix; this is
  // a bit like A(b:c, d:e) in Matlab.
  // This initializer is against the proper semantics of "const", since
  // SubMatrix can change its contents.  It would be hard to implement
  // a "const-safe" version of this class.
  SubMatrix(const MatrixBase<Real>& T,
            const MatrixIndexT ro,  // row offset, 0 < ro < NumRows()
            const MatrixIndexT r,   // number of rows, r > 0
            const MatrixIndexT co,  // column offset, 0 < co < NumCols()
            const MatrixIndexT c)   // number of columns, c > 0
    : MatrixBase<Real>(T.tensor_.index({Slice(ro, ro+r), Slice(co, co+c)})) {}
};
 
  
} // namespace kaldi

#endif
