---
layout: post
title: Benchmarking tKaldi against Kaldi
categories: []
published: true
command_id: 15
excerpt: How fast is tKaldi's binary executable compared against Kaldi's?
---

## Compiling `compute-kaldi-pitch-feats` binary

I have further extended the Vector / Matrix class implementation so that the `featbin/compute-kaldi-pitch-feats.cc` is compiled. This involved adding methods that are used for I/O, and header files like `compressed-matrix.h` and `sparse-matrix.h`. It turned out that these matrix are not related to BLAS operations, so I could bring them in without any modification.

After I compiled tKaldi's version of `compute-kaldi-pitch-feats`, I did some benchmarks.

Two versions are linked to the same MKL as follow;

```
$ ldd /kaldi/src/featbin/compute-kaldi-pitch-feats
  libkaldi-feat.so      => /kaldi/src/lib/libkaldi-feat.so (0x00007f8fb5d43000)
  libkaldi-util.so      => /kaldi/src/lib/libkaldi-util.so (0x00007f8fb5b07000)
  libkaldi-matrix.so    => /kaldi/src/lib/libkaldi-matrix.so (0x00007f8fb585f000)
  libkaldi-base.so      => /kaldi/src/lib/libkaldi-base.so (0x00007f8fb5651000)
  libkaldi-transform.so => /kaldi/src/lib/libkaldi-transform.so (0x00007f8fb4e04000)
  libkaldi-gmm.so       => /kaldi/src/lib/libkaldi-gmm.so (0x00007f8fae313000)
  libkaldi-tree.so      => /kaldi/src/lib/libkaldi-tree.so (0x00007f8fae0bd000)
  libmkl_intel_lp64.so  => /conda/lib/libmkl_intel_lp64.so (0x00007f8fb3ef6000)
  libmkl_core.so        => /conda/lib/libmkl_core.so (0x00007f8fafb76000)
  libmkl_sequential.so  => /conda/lib/libmkl_sequential.so (0x00007f8fae54f000)
  libpthread.so.0       => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f8fb5432000)
  libstdc++.so.6        => /conda/lib/libstdc++.so.6 (0x00007f8fb6238000)
  libgcc_s.so.1         => /conda/lib/libgcc_s.so.1 (0x00007f8fb6224000)
  libc.so.6             => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f8fb5041000)
  libm.so.6             => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f8fb4a66000)
  libdl.so.2            => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f8fadeb9000)
  linux-vdso.so.1 (0x00007fff5b1c6000)
  /lib64/ld-linux-x86-64.so.2 (0x00007f8fb6191000)

$ ldd /tkaldi/bin/compute-kaldi-pitch-feats
  libtkaldi.so          => /tkaldi/build/lib.linux-x86_64-3.8/tkaldi/libtkaldi.so (0x00007f15b7b35000)
  libtorch.so           => /conda/lib/python3.8/site-packages/torch/lib/libtorch.so (0x00007f15b7921000)
  libtorch_cpu.so       => /conda/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so (0x00007f15b0470000)
  libc10.so             => /conda/lib/python3.8/site-packages/torch/lib/libc10.so (0x00007f15b01d1000)
  libgomp.so.1          => /conda/lib/libgomp.so.1 (0x00007f15b81f3000)
  libmkl_intel_lp64.so  => /conda/lib/libmkl_intel_lp64.so (0x00007f15ae306000)
  libmkl_core.so        => /conda/lib/libmkl_core.so (0x00007f15a86a4000)
  libmkl_gnu_thread.so  => /conda/lib/libmkl_gnu_thread.so (0x00007f15aca24000)
  libpthread.so.0       => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f15affb2000)
  libstdc++.so.6        => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f15afc29000)
  libgcc_s.so.1         => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f15afa11000)
  libc.so.6             => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f15af620000)
  libm.so.6             => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f15af282000)
  libdl.so.2            => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f15af07e000)
  librt.so.1            => /lib/x86_64-linux-gnu/librt.so.1 (0x00007f15aee76000)
  linux-vdso.so.1 (0x00007ffe07dde000)
  /lib64/ld-linux-x86-64.so.2 (0x00007f15b8007000)
```

## Benchmark 1. Wall time

The first benchmark is wall time. The binary executable of tKaldi is very slow so I just ran the binary with `time` command. The command looks like the following.

```bash
export OMP_NUM_THREADS=1
time compute-kaldi-pitch-feats \
    --sample-frequency="${rate}" "scp:${scp_path}" "ark:${ark_path}"
```

I used 10 seconds wav file and batched it 32 times in SCP file, which is prepared as follow.

```bash
sox --bits 16 --rate "${rate}" --null --channels 1 "${audio_path}" synth "${audio_length}" sine 300 vol -10db
: > "${scp_path}"
for i in $(seq 32); do
    printf "%s %s\n" "$i" "${audio_path}" >> "${scp_path}"
done
```

With this, I got the following result.

<center>
<table style="">
  <tr>
    <th></th>
    <th>Kaldi</th>
    <th>tKaldi</th>
  </tr>
  <tr>
    <td style="text-align: center">real</td>
    <td style="text-align: right">0.959s</td>
    <td style="text-align: right">50.090s</td>
  </tr>
  <tr>
    <td style="text-align: center">user</td>
    <td style="text-align: right">0.907s</td>
    <td style="text-align: right">49.989s</td>
  </tr>
  <tr>
    <td style="text-align: center">sys</td>
    <td style="text-align: right">0.052s</td>
    <td style="text-align: right">0.064s</td>
  </tr>
</table>
</center>

Well, it's very slow. Very very slow. I was hoping that tKladi's performance is only a few times slower than Kaldi's version, but the reality is very hard.

## Benchmark 2. Flamegraph

Next, I ran `perf` command to get an insight of time consumption.

```bash
perf record --all-cpus --freq=99 --call-graph dwarf \
    compute-kaldi-pitch-feats --sample-frequency="${rate}" "scp:${scp_path}" "ark:${ark_path}"
```

The following figures are the [flame graphs](http://www.brendangregg.com/flamegraphs.html) of these commands. You can click the each component and it will zoom up.

{::nomarkdown}
<embed src="../assets/2020-11-30-benchmarking-tkaldi-against-kaldi/kaldi.svg" style="position: relative; width: 100%;"/>
{:/}

{::nomarkdown}
<embed src="../assets/2020-11-30-benchmarking-tkaldi-against-kaldi/tkaldi.svg" style="position: relative; width: 100%;"/>
{:/}

In general, Torch adds a lot of overheads before the actual computation happens. We might be able to get rid of some of them (like autograd). Let's look into the how the time spent on the core part has changed.

**Kaldi**

<center>
<table>
  <tr>
    <th style="text-align: center" colspan="3">kaldi::OnlinePitchFeatureImpl::AcceptWaveform (1343 samples)</th>
  </tr>
  <tr><td>29.5%</td><td>396 samples</td><td>kaldi::PitchFrameInfo::ComputeBacktraces</td></tr>
  <tr><td>25.9%</td><td>348 samples</td><td>kaldi::ComputeNCCF</td></tr>
  <tr><td>21.9%</td><td>294 samples</td><td>kaldi::ArbitraryResample::Resample</td></tr>
  <tr><td>12.8%</td><td>172 samples</td><td>kaldi::ComputeCorrelation</td></tr>
  <tr><td>3.6%</td><td>48 samples</td><td>kaldi::LinearResample::Resample</td></tr>
</table>
</center>

**tKaldi**

<center>
<table>
  <tr>
    <th style="text-align: center" colspan="3">kaldi::OnlinePitchFeatureImpl::AcceptWaveform (908 samples)</th>
  </tr>
  <tr><td>32.2%</td><td>293 samples</td><td>kaldi::SetNccfPoV</td></tr>
  <tr><td>24.6%</td><td>223 samples</td><td>kaldi::ComputeNCCF</td></tr>
  <tr><td>24.4%</td><td>222 samples</td><td>kaldi::ComputeCorrelation</td></tr>
  <tr><td>8.9%</td><td>81 samples</td><td>kaldi::LinearResample::Resample</td></tr>
  <tr><td>2.2%</td><td>20 samples</td><td>kaldi::PitchFrameInfo::ComputeBacktraces</td></tr>
  <tr><td>1.8%</td><td>16 samples</td><td>kaldi::ArbitraryResample::Resample</td></tr>
</table>
</center>


The original Kaldi implementation spends one-third of time in `PitchFrameInfo::ComputeBacktraces` [[doc](https://kaldi-asr.org/doc/classkaldi_1_1PitchFrameInfo.html#afbc3efb81375265bea318db6855f7f8f), [impl](https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/feat/pitch-functions.cc#L306-L484)], which performs Viterbi computation.

On the other hand, tKaldi's implementation spend much more time on `SetNccfPoV` and `ComputeNCCF` and these functions spend most of time in `at::Tensor::index`. So one of the bottlenecks of tKaldi is indexing. This confirms the intuition I got when I implemented the corresponding <a href="/tkaldi/implementing-kaldis-vector-matrix-library/#element-access-and-memory-access">element access operation</a>.

The next step is to make the tKaldi's computation faster. There are two approaches I can think of so far.

1. Change the highlevel implementation (like `LinearResample` or `OnlinePitchFeatureImpl`) to vectorize the computation. Get rid of element access.

2. Adopt parallel / concurrent model.

Both should work for computations like resampling, but at the moment I am not sure if such approach works for Viterbi algorithm.