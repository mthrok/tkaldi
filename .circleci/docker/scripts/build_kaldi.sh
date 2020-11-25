#!/usr/bin/env bash

# Build Kaldi using MKL from conda installation.

set -ex

apt update -q
apt install -q -y \
    autoconf \
    automake \
    bzip2 \
    g++ \
    gfortran \
    git \
    libtool \
    make \
    python2.7 \
    sox \
    subversion \
    unzip \
    zlib1g-dev
rm -rf /var/lib/apt/lists/*

conda install mkl mkl-include

git clone https://github.com/kaldi-asr/kaldi.git "${KALDI_DIR}"

cd "${KALDI_DIR}"
git checkout "${KALDI_COMMIT}"

cd "${KALDI_DIR}/tools"
make -j "$(nproc)"

cd "${KALDI_DIR}/src"
./configure --shared --mathlib=MKL --mkl-root="${CONDA_PREFIX}" --use-cuda=no
make featbin -j "$(nproc)"
