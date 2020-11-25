#include <torch/script.h>
#include "base/kaldi-types.h"
#include "feat/resample.h"
#include "feat/pitch-functions.h"

using BaseFloat = kaldi::BaseFloat;
using int32 = kaldi::int32;

namespace tkaldi {

  torch::Tensor ResampleWaveform(const torch::Tensor &wave, double orig_freq, double new_freq) {
    kaldi::VectorBase<kaldi::BaseFloat> input(wave);
    kaldi::Vector<kaldi::BaseFloat> output;
    kaldi::ResampleWaveform(orig_freq, input, new_freq, &output);
    return output.tensor_;
  }

  torch::Tensor ComputeKaldiPitch(
      const torch::Tensor &wave,
      double sample_frequency,
      double frame_length,
      double frame_shift,
      double preemphasis_coefficient,
      double min_f0,
      double max_f0,
      double soft_min_f0,
      double penalty_factor,
      double lowpass_cutoff,
      double resample_frequency,
      double delta_pitch,
      double nccf_ballast,
      int64_t lowpass_filter_width,
      int64_t upsample_filter_width,
      int64_t max_frames_latency,
      int64_t frames_per_chunk,
      bool simulate_first_pass_online,
      int64_t recompute_frame,
      bool nccf_ballast_online,
      bool snip_edges
  ) {
    kaldi::VectorBase<kaldi::BaseFloat> input(wave);
    kaldi::PitchExtractionOptions opts;
    opts.samp_freq = static_cast<BaseFloat>(sample_frequency);
    opts.frame_shift_ms = static_cast<BaseFloat>(frame_shift);
    opts.frame_length_ms = static_cast<BaseFloat>(frame_length);
    opts.preemph_coeff = static_cast<BaseFloat>(preemphasis_coefficient);
    opts.min_f0 = static_cast<BaseFloat>(min_f0);
    opts.max_f0 = static_cast<BaseFloat>(max_f0);
    opts.soft_min_f0 = static_cast<BaseFloat>(soft_min_f0);
    opts.penalty_factor = static_cast<BaseFloat>(penalty_factor);
    opts.lowpass_cutoff = static_cast<BaseFloat>(lowpass_cutoff);
    opts.resample_freq = static_cast<BaseFloat>(resample_frequency);
    opts.delta_pitch = static_cast<BaseFloat>(delta_pitch);
    opts.nccf_ballast = static_cast<BaseFloat>(nccf_ballast);
    opts.lowpass_filter_width = static_cast<int32>(lowpass_filter_width);
    opts.upsample_filter_width = static_cast<int32>(upsample_filter_width);
    opts.max_frames_latency = static_cast<int32>(max_frames_latency);
    opts.frames_per_chunk = static_cast<int32>(frames_per_chunk);
    opts.simulate_first_pass_online = simulate_first_pass_online;
    opts.recompute_frame = static_cast<int32>(recompute_frame);
    opts.nccf_ballast_online = nccf_ballast_online;
    opts.snip_edges = snip_edges;
    kaldi::Matrix<kaldi::BaseFloat> output;
    kaldi::ComputeKaldiPitch(opts, input, &output);
    return output.tensor_;
  }

} // namespace tkaldi

TORCH_LIBRARY(tkaldi, m) {
  m.def("tkaldi::ResampleWaveform", &tkaldi::ResampleWaveform);
  m.def("tkaldi::ComputeKaldiPitch", &tkaldi::ComputeKaldiPitch);
}
