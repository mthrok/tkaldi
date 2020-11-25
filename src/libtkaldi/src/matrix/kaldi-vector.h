// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h

#ifndef KALDI_MATRIX_KALDI_VECTOR_H_
#define KALDI_MATRIX_KALDI_VECTOR_H_

#include <torch/script.h>
#include <matrix/matrix-common.h>

using namespace torch::indexing;

namespace kaldi {

namespace {

template<typename Real>
void assert_vector_shape(const torch::Tensor &tensor_) {
  TORCH_INTERNAL_ASSERT(tensor_.ndimension() == 1);
  static_assert(std::is_same<Real, float>::value, "Vector class only supports float32.");
  TORCH_INTERNAL_ASSERT(tensor_.dtype() == torch::kFloat32);
}

} // namespace

template<typename Real> struct MatrixBase;
 
template<typename Real>
struct VectorBase {
  ////////////////////////////////////////////////////////////////////////////////
  // PyTorch-specific things
  ////////////////////////////////////////////////////////////////////////////////
  torch::Tensor tensor_;

  /// Construct VectorBase which is an interface to an existing torch::Tensor object.
  VectorBase(torch::Tensor tensor) : tensor_(tensor) {
    assert_vector_shape<Real>(tensor_);
  };

  ////////////////////////////////////////////////////////////////////////////////
  // Kaldi-compatible methods
  ////////////////////////////////////////////////////////////////////////////////
  /// Empty initializer, corresponds to vector of zero size.
  explicit VectorBase() : tensor_(torch::empty({0})) {
    assert_vector_shape<Real>(tensor_);
  };

  /// Returns the  dimension of the vector.
  inline MatrixIndexT Dim() const { return tensor_.numel(); };

  /// Indexing  operator (const).
  Real operator() (MatrixIndexT i) const {
    return tensor_.index({i}).item().to<Real>();
  };

  /// Indexing operator (non-const).
  Real& operator() (MatrixIndexT i) {
    // CPU only
    return tensor_.accessor<Real, 1>()[i];
  };

  /// Set vector to a specified size (can be zero).
  /// The value of the new data depends on resize_type:
  ///   -if kSetZero, the new data will be zero
  ///   -if kUndefined, the new data will be undefined
  ///   -if kCopyData, the new data will be the same as the old data in any
  ///      shared positions, and zero elsewhere.
  /// This function takes time proportional to the number of data elements.
  void Resize(MatrixIndexT length, MatrixResizeType resize_type = kSetZero) {
    switch(resize_type) {
    case kSetZero:
      tensor_ = torch::zeros({length});
      break;
    case kUndefined:
      tensor_ = torch::empty({length});
      break;
    case kCopyData:
      auto t = torch::zeros({length});
      auto numel = length > tensor_.numel() ? tensor_.numel() : length;
      t.index_put_({Slice(None, numel)}, tensor_.index({Slice(None, numel)}));
      tensor_ = t;
      break;
    }
  }

  /// Add matrix times vector : this <-- beta*this + alpha*M*v.
  /// Calls BLAS GEMV.
  void AddMatVec(const Real alpha, const MatrixBase<Real> &M,
                 const MatrixTransposeType trans,  const VectorBase<Real> &v,
                 const Real beta) { // **beta previously defaulted to 0.0**
    auto mat = M.tensor_;
    if (trans == kTrans) {
      mat = mat.transpose(1, 0);
    }
    tensor_.addmv(mat, v.tensor_, beta, alpha);
  }
};

template<typename Real>
struct Vector : VectorBase<Real> {
  ////////////////////////////////////////////////////////////////////////////////
  // PyTorch-compatibility things
  ////////////////////////////////////////////////////////////////////////////////
  /// Construct VectorBase which is an interface to an existing torch::Tensor object.
  Vector(torch::Tensor tensor) : VectorBase<Real>(tensor) {};

  ////////////////////////////////////////////////////////////////////////////////
  // Kaldi-compatible methods
  ////////////////////////////////////////////////////////////////////////////////
  /// Constructor that takes no arguments.  Initializes to empty.
  Vector(): VectorBase<Real>() {};

  /// Constructor with specific size.  Sets to all-zero by default
  /// if set_zero == false, memory contents are undefined.
  explicit Vector(const MatrixIndexT s,
                  MatrixResizeType resize_type = kSetZero)
      : VectorBase<Real>() {  VectorBase<Real>::Resize(s, resize_type);  }
};

template<typename Real>
struct SubVector : VectorBase<Real> {
  /// Constructor from a Vector or SubVector.
  /// SubVectors are not const-safe and it's very hard to make them
  /// so for now we just give up.  This function contains const_cast.
  SubVector(const VectorBase<Real> &t, const MatrixIndexT origin,
            const MatrixIndexT length)
    : VectorBase<Real>(t.tensor_.index({Slice(origin, origin + length)}))
    {}
};

/// Returns dot product between v1 and v2.
template<typename Real>
Real VecVec(const VectorBase<Real> &v1, const VectorBase<Real> &v2) {
  return torch::dot(v1.tensor_, v2.tensor_).item().template to<Real>();
}

} // namespace kaldi

#endif
