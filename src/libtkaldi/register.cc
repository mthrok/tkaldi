#include <torch/script.h>
#include "base/kaldi-types.h"
#include "feat/resample.h"

namespace tkaldi {

  torch::Tensor ResampleWaveform(const torch::Tensor &wave, double orig_freq, double new_freq) {
    kaldi::VectorBase<kaldi::BaseFloat> input(wave);
    kaldi::Vector<kaldi::BaseFloat> output;
    kaldi::ResampleWaveform(orig_freq, input, new_freq, &output);
    return output.tensor_;
  }

} // namespace tkaldi

TORCH_LIBRARY(tkaldi, m) {
  m.def("tkaldi::ResampleWaveform", &tkaldi::ResampleWaveform);
}
