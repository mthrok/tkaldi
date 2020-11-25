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
  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L42-L43
  void SetZero() { Set(0); }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L48-L49
  void Set(Real f) { tensor_.index_put_({"..."}, f); }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L62-L63
  inline MatrixIndexT Dim() const { return tensor_.numel(); };

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L68-L69
  inline Real* Data() { return tensor_.data_ptr<Real>(); }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L71-L72
  inline const Real* Data() const { return tensor_.data_ptr<Real>(); }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L74-L79
  inline Real operator() (MatrixIndexT i) const {
    return tensor_.index({i}).item().to<Real>();
  };

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L81-L86
  inline Real& operator() (MatrixIndexT i) {
    // CPU only
    return tensor_.accessor<Real, 1>()[i];
  };

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L88-L95
  SubVector<Real> Range(const MatrixIndexT o, const MatrixIndexT l) {
    return SubVector<Real>(*this, o, l);
  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L97-L105
  const SubVector<Real> Range(const MatrixIndexT o,
                              const MatrixIndexT l) const {
    return SubVector<Real>(*this, o, l);
  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L107-L108
  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.cc#L226-L233
  void CopyFromVec(const VectorBase<Real> &v) {
    TORCH_INTERNAL_ASSERT(tensor_.sizes() == v.tensor_.sizes());
    tensor_.copy_(v.tensor_);
  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L181-L184
  template<typename OtherReal>
  void AddVec(const Real alpha, const VectorBase<OtherReal> &v) {
    tensor_ += alpha * v.tensor_;
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

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L233-L234
  void Add(Real c) { tensor_ += c; }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L236-L239
  void AddVecVec(Real alpha, const VectorBase<Real> &v,
                 const VectorBase<Real> &r, Real beta) {
    tensor_ = beta * tensor_ + alpha * v.tensor_ * r.tensor_;
  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L246-L247
  void Scale(Real alpha) { tensor_ *= alpha; }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L305-L306
  Real Min() const {
    if (tensor_.numel()) {
      return tensor_.min().item().to<Real>();
    }
    return std::numeric_limits<Real>::infinity();
  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L308-L310
  Real Min(MatrixIndexT *index) const {
    TORCH_INTERNAL_ASSERT(tensor_.numel());
    torch::Tensor value, ind;
    std::tie(value, ind) = tensor_.min(0);
    *index = ind.item().to<MatrixIndexT>();
    return value.item().to<Real>();
  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L312-L313
  Real Sum() const { return tensor_.sum().item().to<Real>(); };

protected:
  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L362-L365
  explicit VectorBase() : tensor_(torch::empty({0})) {
    assert_vector_shape<Real>(tensor_);
  };
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
      : VectorBase<Real>() {  Resize(s, resize_type);  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L412-L416
  explicit Vector(const VectorBase<Real> &v) : VectorBase<Real>() {
    Resize(v.Dim(), kUndefined);
    this->CopyFromVec(v);
  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L434-L435
  void Swap(Vector<Real> *other) {
    auto tmp = VectorBase<Real>::tensor_;
    this->tensor_ = other->tensor_;
    other->tensor_ = tmp;
  }

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L444-L451
  void Resize(MatrixIndexT length, MatrixResizeType resize_type = kSetZero) {
    // Note: The original Kaldi implementation uses in-place resizing, but here, a new Tensor is allocated.
    // TODO: Check the difference.
    auto &tensor_ = VectorBase<Real>::tensor_;
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

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L463-L468
  Vector<Real> &operator = (const VectorBase<Real> &other) {
    Resize(other.Dim(), kUndefined);
    this->CopyFromVec(other);
    return *this;
  }
};

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L482-L485
template<typename Real>
struct SubVector : VectorBase<Real> {
  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L487-L499
  SubVector(const VectorBase<Real> &t, const MatrixIndexT origin,
            const MatrixIndexT length)
    : VectorBase<Real>(t.tensor_.index({Slice(origin, origin + length)}))
    {}

  // https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L524-L528
  SubVector(const MatrixBase<Real> &matrix, MatrixIndexT row)
    : VectorBase<Real>(matrix.tensor_.index({Slice(row), Slice()}))
    {}
};

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-vector.h#L573-L575
template<typename Real>
Real VecVec(const VectorBase<Real> &v1, const VectorBase<Real> &v2) {
  return torch::dot(v1.tensor_, v2.tensor_).item().template to<Real>();
}

} // namespace kaldi

#endif
