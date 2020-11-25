#!/usr/bin/env bash

set -euo pipefail

cd "$( dirname "${BASH_SOURCE[0]}" )"

image="mthrok/tkaldi-test-base"
kaldi_commit="0c6a3dcf0"
datestr="$(date +"%Y-%m-%d")"

for python_version in 3.8 3.7 3.6; do
    tag="py${python_version}-${kaldi_commit}-${datestr}"
    docker build \
           --build-arg PYTHON_VERSION="${python_version}" \
           --build-arg KALDI_COMMIT="${kaldi_commit}" \
           -t"${image}:${tag}" .
done
