---
layout: post
title: Implementing Kaldi's Vector/Matrix library
categories: []
published: true
comment_id: 4
excerpt: Implements Kaldi's Vector/Matrix classes with PyTorch's Tensor class.
---

The basis of Kaldi is the family of vector / matrix classes, which are backed by a low level BLAS library.

Implementing the same interface with PyTorch's Tensor class should not be difficult. PyTorch's Tensor class takes care of storage, dispatching and vectorization, in addition to vector / matrix arithmetics.

## Components

Firstly, let's figure out which methods are needed for what. Let's take `ComputeKaldiPitch`[[doc](https://kaldi-asr.org/doc/group__feat.html#gab2d0682a863b69133865d08ff795f7dc), [source](https://github.com/kaldi-asr/kaldi/blob/d79c896b444a3ba3418b0e2387676d6209bf43fe/src/feat/pitch-functions.cc#L1291-L1327)] as an example. This function has the following internal dependencies.

* `OnlinePitchFeature`[[doc](https://kaldi-asr.org/doc/classkaldi_1_1OnlinePitchFeature.html), [source](https://github.com/kaldi-asr/kaldi/blob/d79c896b444a3ba3418b0e2387676d6209bf43fe/src/feat/pitch-functions.h#L298-L325)]
   * `OnlinePitchFeatureImpl`[[doc](https://kaldi-asr.org/doc/classkaldi_1_1OnlinePitchFeatureImpl.html), [source](https://github.com/kaldi-asr/kaldi/blob/d79c896b444a3ba3418b0e2387676d6209bf43fe/src/feat/pitch-functions.cc#L572-L712)]
       * `ArbitraryResample`[[doc](https://kaldi-asr.org/doc/classkaldi_1_1ArbitraryResample.html), [source](https://github.com/kaldi-asr/kaldi/blob/d79c896b444a3ba3418b0e2387676d6209bf43fe/src/feat/resample.h#L87-L134)]
       * `LinearResample`[[doc](https://kaldi-asr.org/doc/classkaldi_1_1LinearResample.html), [source](https://github.com/kaldi-asr/kaldi/blob/d79c896b444a3ba3418b0e2387676d6209bf43fe/src/feat/resample.h#L137-L253)]

Looking at these interfaces, the following classes are used.

* Vector Classes
    * `VectorBase`[[doc](https://kaldi-asr.org/doc/classkaldi_1_1VectorBase.html), [source](https://github.com/kaldi-asr/kaldi/blob/d79c896b444a3ba3418b0e2387676d6209bf43fe/src/matrix/kaldi-vector.h#L37-L399)]
    * `Vector`[[doc](https://kaldi-asr.org/doc/classkaldi_1_1Vector.html), [source](https://github.com/kaldi-asr/kaldi/blob/d79c896b444a3ba3418b0e2387676d6209bf43fe/src/matrix/kaldi-vector.h#L401-L495)]
    * `SubVector` [[doc](https://kaldi-asr.org/doc/classkaldi_1_1SubVector.html), [source](https://github.com/kaldi-asr/kaldi/blob/d79c896b444a3ba3418b0e2387676d6209bf43fe/src/matrix/kaldi-vector.h#L498-L550)]

* Matrix Classes
    * `MatrixBase`[[doc](https://kaldi-asr.org/doc/classkaldi_1_1MatrixBase.html), [source](https://github.com/kaldi-asr/kaldi/blob/d79c896b444a3ba3418b0e2387676d6209bf43fe/src/matrix/kaldi-matrix.h#L45-L820)]
    * `Matrix`[[doc](https://kaldi-asr.org/doc/classkaldi_1_1Matrix.html), [source](https://github.com/kaldi-asr/kaldi/blob/d79c896b444a3ba3418b0e2387676d6209bf43fe/src/matrix/kaldi-matrix.h#L822-L948)]
    * `SubMatrix`[[doc](https://kaldi-asr.org/doc/classkaldi_1_1SubMatrix.html), [source](https://github.com/kaldi-asr/kaldi/blob/d79c896b444a3ba3418b0e2387676d6209bf43fe/src/matrix/kaldi-matrix.h#L981-L1020)]

There are other variations like `Sparse`, `Compressed`, `Packed`, `Sp` and `Tp`, then CUDA versions of them, but for now, we can forget about them.

## Implementation

`VectorBase` / `MatrixBase` classes implement all the algebra operatoins. On top of that, the plain `Vector` / `Matrix` classes handles memory allocations. `SubVector` / `SubMatrix` classes are the representation of sliced objects.

Since Tensor class comes with memory management, we do not need to manage memory by ourselves. All these classes can contain internal reference to a Tensor object and all the operations can be applied to the refernce object. So a constructor would look like this;

```c++
template<typename Real>
struct VectorBase {
  torch::Tensor tensor_;

  VectorBase(torch::Tensor tensor) : tensor_(tensor) {
    TORCH_INTERNAL_ASSERT(tensor_.ndimension() == 1);
  };

  ...

  inline MatrixIndexT Dim() const { return tensor_.numel(); };

  ...
}
```

Since the `Vector` classes represent 1D array, the input `Tensor` has to be one dimensional. So in the above code snippet, the shape of the `Tensor` is validated to be one-dimentional. With this validatoin, `VectorBase::Dim` method can be written using `Tensor::nueml`. Note that `Vector::Dim` and `Tensor::ndimension` are totally different.

### Element Access and Memory Access

Kaldi's `Vector` / `Matrix` classes use `operator() (MatrixIndexT i)` for element access. There are [`const` version](https://github.com/kaldi-asr/kaldi/blob/d79c896b444a3ba3418b0e2387676d6209bf43fe/src/matrix/kaldi-vector.h#L75-L80) and [`non-const` version](https://github.com/kaldi-asr/kaldi/blob/d79c896b444a3ba3418b0e2387676d6209bf43fe/src/matrix/kaldi-vector.h#L82-L87). `non-const` version is used to write an element ([example](https://github.com/kaldi-asr/kaldi/blob/d79c896b444a3ba3418b0e2387676d6209bf43fe/src/feat/resample.cc#L309)).

On the other hand, PyTorch's Tensor class adopted functional form for element access for [both read/write](https://pytorch.org/cppdocs/notes/tensor_indexing.html#tensor-indexing-api). I am guessing that this is due to the fact that a Tensor object might be on CUDA device, which does not provide a way to access the memory similar to regular CPU memory.

The following snippet illustrates a naive way to implement `operator()`.

```c++
  inline Real operator() (MatrixIndexT i) const {
    return tensor_.index({i}).item().to<Real>();
  };

  inline Real& operator() (MatrixIndexT i) {
    return tensor_.accessor<Real, 1>()[i];
  };
```

This snippet is very interesting, because it is obvious that these ways of accessing an element are very inefficient. For the read operation, it goes through three operations (indexing, extraction, then conversion). For the write operation, this implementation only works for CPU Tensors. I would revisit this once I get to the point that my implementation produces the numerically same results. However I get the feeling that the right way is to rewrite the client code to avoid element-wise access and vectorize the memory access.

### Linear Algebra

Unlike these memory-related issues, implementing algebraic operations are straightforward. Operations that are wrapper around the underlying BLAS functions have one-to-one mapping between Kaldi and PyTorch. `AddMatVec` is a good example of this. 

```c++
  void AddMatVec(const Real alpha, const MatrixBase<Real> &M,
                 const MatrixTransposeType trans,  const VectorBase<Real> &v,
                 const Real beta) { // **beta previously defaulted to 0.0**
    auto mat = M.tensor_;
    if (trans == kTrans) {
      mat = mat.transpose(1, 0);
    }
    tensor_.addmv_(mat, v.tensor_, beta, alpha);
  }
```

### SubMatrix / SubVector as sliced Tensor

Implementing `SubVector` / `SubMatrix` is easy too. All you need is to slice the original `Tensor` object properly.

```c++
template<typename Real>
struct SubVector : VectorBase<Real> {

  SubVector(const VectorBase<Real> &t, const MatrixIndexT origin,
            const MatrixIndexT length)
    : VectorBase<Real>(t.tensor_.index({Slice(origin, origin + length)}))
    {}
```

You can checkout the implementation of `matrix/kaldi-vector` and `matrix/kaldi-matrix` modules from [here](https://github.com/mthrok/tkaldi/tree/main/src/libtkaldi/src/matrix). Note that it is a work in progress.