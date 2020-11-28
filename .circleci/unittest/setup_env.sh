#!/usr/bin/env bash

set -ex

conda install pytorch cpuonly -c pytorch

conda install pytest scipy parameterized
pip install kaldi_io cmake
