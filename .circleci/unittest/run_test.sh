#!/usr/bin/env bash

set -eux

python -m torch.utils.collect_env

pytest tests/tkaldi_unittest -v
