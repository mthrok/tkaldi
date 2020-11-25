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

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L99-L107
  inline Real&  operator() (MatrixIndexT r, MatrixIndexT c) {
    // CPU only
    return tensor_.accessor<Real, 2>()[r][c];
  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L112-L120
  inline const Real operator() (MatrixIndexT r, MatrixIndexT c) const {
    return tensor_.index({Slice(r), Slice(c)}).item().to<Real>();
  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L138-L141
  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.cc#L859-L898
  template<typename OtherReal>
  void CopyFromMat(const MatrixBase<OtherReal> & M,
                   MatrixTransposeType trans = kNoTrans) {
    auto src = M.tensor_;
    if (trans == kTrans)
      src = src.transpose(1, 0);
    tensor_.index_put_({Slice(), Slice()}, src);
  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L186-L191
  inline const SubVector<Real> Row(MatrixIndexT i) const {
    return SubVector<Real>(*this, i);
  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L208-L211
  inline SubMatrix<Real> RowRange(const MatrixIndexT row_offset,
                                  const MatrixIndexT num_rows) const {
    return SubMatrix<Real>(*this, row_offset, num_rows, 0, NumCols());
  }

protected:

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L749-L753
  explicit MatrixBase(): tensor_(torch::empty({0, 0})) {
    KALDI_ASSERT_IS_FLOATING_TYPE(Real);
  }
};

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L781-L784
template<typename Real>
struct Matrix : MatrixBase<Real> {
  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L786-L787
  Matrix() : MatrixBase<Real>() {}

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L789-L793
 Matrix(const MatrixIndexT r, const MatrixIndexT c,
        MatrixResizeType resize_type = kSetZero,
        MatrixStrideType stride_type = kDefaultStride):
      MatrixBase<Real>() { Resize(r, c, resize_type, stride_type); }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L859-L874
  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.cc#L817-L857
  void Resize(const MatrixIndexT r,
              const MatrixIndexT c,
              MatrixResizeType resize_type = kSetZero,
              MatrixStrideType stride_type = kDefaultStride) {
    // Note: The original Kaldi implementation uses in-place resizing, but here, a new Tensor is allocated.
    // TODO: Check the difference.
    auto &tensor_ = MatrixBase<Real>::tensor_;
    switch(resize_type) {
    case kSetZero:
      tensor_ = torch::zeros({r, c});
      break;
    case kUndefined:
      tensor_ = torch::empty({r, c});
      break;
    case kCopyData:
      auto t = torch::zeros({r, c});
      auto num_row = r < tensor_.size(0) ? r : tensor_.size(0);
      auto num_col = c < tensor_.size(1) ? c : tensor_.size(1);
      t.index_put_({Slice(None, num_row), Slice(None, num_col)}, tensor_.index({Slice(None, num_row), Slice(None, num_col)}));
      tensor_ = t;
      break;
    }
  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L876-L883
  Matrix<Real> &operator = (const MatrixBase<Real> &other) {
    if (MatrixBase<Real>::NumRows() != other.NumRows() ||
        MatrixBase<Real>::NumCols() != other.NumCols())
      Resize(other.NumRows(), other.NumCols(), kUndefined);
    MatrixBase<Real>::CopyFromMat(other);
    return *this;
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
