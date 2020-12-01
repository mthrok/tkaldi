// matrix/kaldi-matrix.h

// Copyright 2009-2011  Ondrej Glembek;  Microsoft Corporation;  Lukas Burget;
//                      Saarland University;  Petr Schwarz;  Yanmin Qian;
//                      Karel Vesely;  Go Vivace Inc.;  Haihua Xu
//           2017       Shiyin Kang

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

// Based on https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h

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
    return tensor_.accessor<Real, 2>()[r][c];
  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L124-L125
  void SetZero() { tensor_.zero_(); }

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

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L200-L207
  inline SubMatrix<Real> Range(const MatrixIndexT row_offset,
                               const MatrixIndexT num_rows,
                               const MatrixIndexT col_offset,
                               const MatrixIndexT num_cols) const {
    return SubMatrix<Real>(*this, row_offset, num_rows,
                           col_offset, num_cols);
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

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L235-L236
  void Scale(Real alpha) { tensor_ *= alpha; }

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

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L802-L803
  void Swap(Matrix<Real> *other) {
    auto tmp = this->tensor_;
    this->tensor_ = other->tensor_;
    other->tensor_ = tmp;
  }

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

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L829-L830
  explicit Matrix(const CompressedMatrix &C);

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

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L913-L924
struct HtkHeader {
  /// Number of samples.
  int32    mNSamples;
  /// Sample period.
  int32    mSamplePeriod;
  /// Sample size
  int16    mSampleSize;
  /// Sample kind.
  uint16   mSampleKind;
};

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L926-L928
template<typename Real>
bool ReadHtk(std::istream &is, Matrix<Real> *M, HtkHeader *header_ptr);

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L930-L932
template<typename Real>
bool WriteHtk(std::ostream &os, const MatrixBase<Real> &M, HtkHeader htk_hdr);
 
// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L940-L948
template<typename Real>
struct SubMatrix : MatrixBase<Real> {
  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L950-L959
  SubMatrix(const MatrixBase<Real>& T,
            const MatrixIndexT ro,  // row offset, 0 < ro < NumRows()
            const MatrixIndexT r,   // number of rows, r > 0
            const MatrixIndexT co,  // column offset, 0 < co < NumCols()
            const MatrixIndexT c)   // number of columns, c > 0
    : MatrixBase<Real>(T.tensor_.index({Slice(ro, ro+r), Slice(co, co+c)}))
    {}

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L961-L966
  SubMatrix(Real *data,
            MatrixIndexT num_rows,
            MatrixIndexT num_cols,
            MatrixIndexT stride)
    : MatrixBase<Real>(torch::from_blob(data, {num_rows, num_cols}, {stride, 1}))
    {}
};

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.h#L1059-L1060
template<typename Real>
std::ostream & operator << (std::ostream & Out, const MatrixBase<Real> & M) {
  Out << M.tensor_;
  return Out;
}
  
} // namespace kaldi

#endif
