// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h

#ifndef KALDI_MATRIX_KALDI_VECTOR_H_
#define KALDI_MATRIX_KALDI_VECTOR_H_

#include <torch/torch.h>
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

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L36-L40
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
  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L362-L365
  explicit VectorBase() : tensor_(torch::empty({0})) {
    assert_vector_shape<Real>(tensor_);
  };

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L62-L63
  inline MatrixIndexT Dim() const { return tensor_.numel(); };

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L74-L79
  inline Real operator() (MatrixIndexT i) const {
    return tensor_.index({i}).item().to<Real>();
  };

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L81-L86
  inline Real& operator() (MatrixIndexT i) {
    // CPU only
    return tensor_.accessor<Real, 1>()[i];
  };

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L444-L451
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

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L196-L198
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

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L385-L390
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
  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L392-L393
  Vector(): VectorBase<Real>() {};

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L395-L399
  explicit Vector(const MatrixIndexT s,
                  MatrixResizeType resize_type = kSetZero)
      : VectorBase<Real>() {  VectorBase<Real>::Resize(s, resize_type);  }
};

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L482-L485
template<typename Real>
struct SubVector : VectorBase<Real> {
  SubVector(const VectorBase<Real> &t, const MatrixIndexT origin,
            const MatrixIndexT length)
    : VectorBase<Real>(t.tensor_.index({Slice(origin, origin + length)}))
    {}
};

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L573-L575
template<typename Real>
Real VecVec(const VectorBase<Real> &v1, const VectorBase<Real> &v2) {
  return torch::dot(v1.tensor_, v2.tensor_).item().template to<Real>();
}

} // namespace kaldi

#endif
