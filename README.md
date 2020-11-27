[![CircleCI](https://circleci.com/gh/mthrok/tkaldi.svg?style=svg)](https://circleci.com/gh/mthrok/tkaldi?branch=main)

# tKaldi

Yet Another Aproach to Port Kaldi

This is an experimental attempt to re-write Kaldi's matrix library with PyTorch's C++ API.

Note: This is my Sunday project. 

## Approach to Port Kaldi

This project aims to implement the following classes as wrppers around
PyTorch's `torch::Tensor` class.

**Vector Classes**
 - `kaldi::VectorBase`
 - `kaldi::Vector`
 - `kaldi::SubVector`

**Matrix Classes**
 - `kaldi::MatrixBase`
 - `kaldi::Matrix`
 - `kaldi::SubMatrix`

(You can check out the code from [here](./src/libtkaldi/src).)

Theoretically, by swapping the original source codes with these implementations,
we should be able to build the reset of Kaldi libraries.
(Except the parts related to CUDA and OpenFST, which I have not looked into.)

Once we build the Kaldi code with PyTorch's backend, it should be fairly easy to
build the PyTorch binding of the resulting library, and this means that we can call
Kaldi functions from PyTorch natively.

## Execution

Since Kaldi's code base is huge, it is difficult to start by forking it and modifying it.
Instead, I took a bottom up approach, which is, deciding on a target feature that I want
to port, and then implementing the necessary interface of Vector/Matrix classes.

When compiling the target feature, the source code of the target features are copied to
the workspace with minimum modification. Interestingly, all I had to do so far was to 
comment out some `#include` statements, which are not directly related to the target feature,
and swapping some type definitions. You can checkout these in [kaldi.patch](./kaldi.patch).

For the initial target feature, I choese [`ComputeKaldiPitch`](https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/feat/pitch-functions.h#L411-L419) and the corresponding CLI, [`compute-kaldi-pitch-feats`](https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/featbin/compute-kaldi-pitch-feats.cc).

I am porting these features in the following manner.

### Phase 1 - Port `ComputeKaldiPitch`

The goal of this phase is to have `ComputeKaldiPitch` function that produces the exact same result 
as the original implementation. The performance of the function does not matter. In fact, since the
resulting Vector / Matrix classes are wrapper around `torch::Tensor`, and `torch::Tensor` is backed
by a similar (or same) BLAS library, while Kaldi's original implementation directly calls the BLAS 
library, it is expected to be slower or at the same speed at best.

- [x] Implement the minimal set of methods from Vector / Matrix classes. [016ab2e7](https://github.com/mthrok/tkaldi/tree/016ab2e7d757ae654607fc60dfceadc2a6c26ada/src/libtkaldi/src/matrix)
- [x] Compile `ComputeKaldiPitch`.
- [x] Bind the resulting `ComputeKaldiPitch` to Python. [src](https://github.com/mthrok/tkaldi/blob/016ab2e7d757ae654607fc60dfceadc2a6c26ada/src/libtkaldi/register.cc#L18-L66)
- [x] Check the parity of the Python function and `compute-kaldi-pitch-feats` from the original code. [test](https://app.circleci.com/pipelines/github/mthrok/tkaldi/50/workflows/d2ba7389-4088-47db-b315-45b3f863c0c3)

### Phase 2 - Port `compute-kaldi-pitch-feats`

The next step is to port `compute-kaldi-pitch-feats` CLI so that I can compare the speed of the 
original CLI and the ported version.

- [x] Extend the Vector / Matrix classes [bc8ac3c0](https://github.com/mthrok/tkaldi/tree/bc8ac3c0e85c4cb08242c837f7ccaf39b49ca619/src/libtkaldi/src/matrix).
- [ ] Compile `compute-kaldi-pitch-feats`
- [ ] Compare the speed of the original `compute-kaldi-pitch-feats` and ported one.

### Phase 3 - Improve the performace of `ComputeKaldiPitch`

The third step is to improve the speed of `ComputeKaldiPitch` by modifying the implementation to take
advantage of PyTorch's C++ API. (and potentially getting rid of Vector / Matrix classes).

- [ ] Vectorize the operation and get rid of sequential element access.
- [ ] Parallelize operations.
- [ ] (Optional) Enable GPU support.

## Build

Because of the approach explained in the previous section, this repository is not a fork of the original Kaldi.
Instead, this repository references Kaldi as `git-submodule` and copy the required source codes from them.

[tools.py](./tools.py) facilitates this process.

**Note** When changing the list of source files under source control in [`src/libtkaldi/src`](./src/libtkaldi/src),
edit [`.gitignore`](.gitignore) and [`tools.py`](./tools.py)

* `./tools.py init`  
This will sync the Kaldi submodule (in [`third_party/kaldi`](./third_party)), clean up the any changes present there,
then apply the patch form [`kaldi.patch`](./kaldi.patch).

* `./tools.py dev`  
This will run `git-clean` on the current [`src/libtkaldi`](./src/libtkaldi) (so that files that are not
under source control will be removed), copy the designated source codes from `third_party/kaldi` directory,
then run `python setup.py develop` to build the library.

* `./tools.py generate_patch` or `./tools.py gen`
This will stash the changes made to Kaldi submodule to [`kaldi.patch`](./kaldi.patch). When you apply change to
the original source code of Kaldi and you need to persist the change, you need to check-in the patch.

### Getting Started

```
git clone https://github.com/mthrok/tkaldi
cd tkaldi
./tools.py init
```

### Building and Runnig test

```
./tools.py dev
pytest tests
```

## Requirements

```
pytorch >= 1.7
```
