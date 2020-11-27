// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h

#ifndef KALDI_MATRIX_KALDI_MATRIX_H_
#define KALDI_MATRIX_KALDI_MATRIX_H_

#include <torch/torch.h>
#include "matrix/matrix-common.h"
#include "matrix/kaldi-vector.h"

using namespace torch::indexing;

namespace kaldi {

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L44-L48
template<typename Real>
struct MatrixBase {
  ////////////////////////////////////////////////////////////////////////////////
  // PyTorch-specific items
  ////////////////////////////////////////////////////////////////////////////////
  torch::Tensor tensor_;
  /// Construct VectorBase which is an interface to an existing torch::Tensor object.
  MatrixBase(torch::Tensor tensor);

  ////////////////////////////////////////////////////////////////////////////////
  // Kaldi-compatible items
  ////////////////////////////////////////////////////////////////////////////////
  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L62-L63
  inline MatrixIndexT NumRows() const { return tensor_.size(0); };

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L65-L66
  inline MatrixIndexT NumCols() const { return tensor_.size(1); };

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L68-L69
  inline MatrixIndexT Stride() const {  return tensor_.stride(0); }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L77-L80
  inline const Real* Data() const { return tensor_.data_ptr<Real>(); }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L82-L83
  inline Real* Data() { return tensor_.data_ptr<Real>(); }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L85-L90
  inline  Real* RowData(MatrixIndexT i) { return tensor_.index({i}).data_ptr<Real>(); }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L92-L97
  inline const Real* RowData(MatrixIndexT i) const { return tensor_.index({i}).data_ptr<Real>(); }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L177-L178
  void CopyColFromVec(const VectorBase<Real> &v, const MatrixIndexT col) {
    tensor_.index_put_({Slice(), col}, v.tensor_);
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

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L224-L225
  Real Max() const { return tensor_.max().item().to<Real>(); }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L226-L227
  Real Min() const {return tensor_.min().item().to<Real>(); }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L567-L569
  void AddMat(const Real alpha, const MatrixBase<Real> &M,
              MatrixTransposeType transA = kNoTrans) {
    tensor_ += alpha * (transA == kNoTrans ? M.tensor_ : M.tensor_.transpose(1, 0));
  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L720-L723
  void Read(std::istream & in, bool binary, bool add = false);

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L724-L725
  void Write(std::ostream & out, bool binary) const;

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L741
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
         MatrixStrideType stride_type = kDefaultStride)
    : MatrixBase<Real>() { Resize(r, c, resize_type, stride_type); }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L808-L811
  explicit Matrix(const MatrixBase<Real> & M,
                  MatrixTransposeType trans = kNoTrans)
    : MatrixBase<Real>(trans == kNoTrans ? M.tensor_ : M.tensor_.transpose(1, 0))
    {}

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L816-L819
  template<typename OtherReal>
  explicit Matrix(const MatrixBase<OtherReal> & M,
                  MatrixTransposeType trans = kNoTrans)
    : MatrixBase<Real>(trans == kNoTrans ? M.tensor_ : M.tensor_.transpose(1, 0))
    {}

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L845-L847
  void Read(std::istream & in, bool binary, bool add = false);

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L859-L874
  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.cc#L817-L857
  void Resize(const MatrixIndexT r,
              const MatrixIndexT c,
              MatrixResizeType resize_type = kSetZero,
              MatrixStrideType stride_type = kDefaultStride) {
    auto &tensor_ = MatrixBase<Real>::tensor_;
    switch(resize_type) {
    case kSetZero:
      tensor_.resize_({r, c}).zero_();
      break;
    case kUndefined:
      tensor_.resize_({r, c});
      break;
    case kCopyData:
      auto tmp = tensor_;
      auto tmp_rows = tmp.size(0);
      auto tmp_cols = tmp.size(1);
      tensor_.resize_({r, c}).zero_();
      auto rows = Slice(None, r < tmp_rows ? r : tmp_rows);
      auto cols = Slice(None, c < tmp_cols ? c : tmp_cols);
      tensor_.index_put_({rows, cols}, tmp.index({rows, cols}));
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

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L1059-L1060
template<typename Real>
std::ostream & operator << (std::ostream & Out, const MatrixBase<Real> & M) {
  Out << M.tensor_;
  return Out;
}
  
} // namespace kaldi

#endif
