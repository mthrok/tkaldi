---
layout: post
title: Benchmarking tKaldi against Kaldi
categories: []
published: false
command_id: 0

---

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

Next, I ran `perf` command to get an insight of time consumption.

```bash
perf record --all-cpus --freq=99 --call-graph dwarf \
    compute-kaldi-pitch-feats --sample-frequency="${rate}" "scp:${scp_path}" "ark:${ark_path}"
```

The following figures are the [flame graphs](http://www.brendangregg.com/flamegraphs.html) of these commands. You can click to zoom up.

{::nomarkdown}
<embed src="../assets/2020-11-30-benchmarking-tkaldi-against-kaldi/kaldi.svg" style="position: relative; width: 100%;"/>
{:/}

{::nomarkdown}
<embed src="../assets/2020-11-30-benchmarking-tkaldi-against-kaldi/tkaldi.svg" style="position: relative; width: 100%;"/>
{:/}

I am new to these performance tools so cannot get much insight from these graphs, but as expected, there is a lot of overhead in tKaldi's version, which I do not think have much control.



In the next step, 
