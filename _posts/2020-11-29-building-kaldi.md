---
layout: post
title: Building Kaldi with custom Matrix implementations
categories: []
published: true
comment_id: 13

---

Once the Vector / Matrix classes are implemented, then we can bring in the implementation of higher level functions. We need to make some modifications to header files, though, so that the implementations missing and not used, like `sp-matrix.h`, are not included. This can be done in [`matrix-lib.h`](https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/matrix-lib.h#L28-L34)

Another modification required is Kaldi's numeric type definitions. Kaldi's numeric type definisions, like `int32`, are from OpenFST, and there is a place to chage [this](https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/base/kaldi-types.h#L57-L73).

With only these two changes, we can bring in the rest of the code for `ComputeKaldiPitch` and they compile fine.

Finally, to make it accessible from Python code, we need a wrapper around `ComputeKaldiPitch`. The code looks like the following [[src](https://github.com/mthrok/tkaldi/blob/016ab2e7d757ae654607fc60dfceadc2a6c26ada/src/libtkaldi/register.cc#L18-L66)]. It just creates `kaldi::VectorBase` object from an existing Tensor and extract the result from `kaldi::Matrix` as a Tensor.

```c++
torch::Tensor ComputeKaldiPitch(
      const torch::Tensor &wave,
      double sample_frequency,
      double frame_length,

      # The rest of the options go here
) {
    kaldi::VectorBase<kaldi::BaseFloat> input(wave);
    kaldi::PitchExtractionOptions opts;
    opts.samp_freq = static_cast<BaseFloat>(sample_frequency);
    opts.frame_length_ms = static_cast<BaseFloat>(frame_length);

    # The rest of the options go...
    ...

    kaldi::Matrix<kaldi::BaseFloat> output;
    kaldi::ComputeKaldiPitch(opts, input, &output);
    return output.tensor_;
}
```

With the proper implementation of Vector/Matrix classes, calling this function produces the exact same result as the original `compute-kaldi-pitch-feats` command. Test code can be found [here](https://github.com/mthrok/tkaldi/blob/016ab2e7d757ae654607fc60dfceadc2a6c26ada/tests/tkaldi_unittest/feats_test.py#L10-L36) and the result can be seen [here](https://app.circleci.com/pipelines/github/mthrok/tkaldi/51/workflows/0043d82b-7fe2-4498-a134-90d6a73efefc/jobs/138/parallel-runs/0/steps/0-104).
