#!/usr/bin/env bash

set -eu

output_dir="$1"
audio_length="$2"
num_repeats="$3"

ROOT_DIR="$(git rev-parse --show-toplevel)"
export PATH="${PATH}:${ROOT_DIR}/third_party/FlameGraph"
export OMP_NUM_THREADS=1

rate=44100

# TODO: compiile with CMAKE_CXX_FLAGS=-fno-omit-frame-pointer ./tools.py develop

WORKDIR="$(mktemp -d)"
cleanup () { rm -rf "${WORKDIR}"; }
trap cleanup EXIT

# Generate files
audio_path="${WORKDIR}/foo.wav"
scp_path="${WORKDIR}/foo.scp"
ark_path="${WORKDIR}/foo.ark"

: > "${scp_path}"
for i in $(seq ${num_repeats}); do
    printf "%s %s\n" "$i" "${audio_path}" >> "${scp_path}"
done
set -x
sox --bits 16 --rate "${rate}" --null --channels 1 "${audio_path}" synth "${audio_length}" sine 300 vol -10db

# Run
which compute-kaldi-pitch-feats

(
    cd "${output_dir}"
    # time compute-kaldi-pitch-feats --sample-frequency="${rate}" "scp:${scp_path}" "ark:${ark_path}"
    # Record
    perf record --all-cpus --freq=99 --call-graph dwarf \
         compute-kaldi-pitch-feats --sample-frequency="${rate}" "scp:${scp_path}" "ark:${ark_path}"
    perf script > perf.txt

    stackcollapse-perf.pl perf.txt > perf.folded.txt
    flamegraph.pl perf.folded.txt > perf.svg
)
