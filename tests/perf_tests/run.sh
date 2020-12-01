#!/usr/bin/env bash

mkdir -p perf/kaldi
./tests/perf_tests/measure.sh perf/kaldi 5 1500

export PATH="${PWD}/src/tkaldi/bin:${PATH}"

mkdir -p perf/tkaldi
./tests/perf_tests/measure.sh perf/tkaldi 5 50
