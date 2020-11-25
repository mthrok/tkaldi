"""Submodule for kaldi's featsbin"""

import torch


def compute_kaldi_pitch(
        wave: torch.Tensor,
        sample_frequency: float,
        frame_length: float = 25.0,
        frame_shift: float = 10.0,
        preemph_coeff: float = 0.0,
        min_f0: float = 50,
        max_f0: float = 400,
        soft_min_f0: float = 10.0,
        penalty_factor: float = 0.1,
        lowpass_cutoff: float = 1000,
        resample_frequency: float = 4000,
        delta_pitch: float = 0.005,
        nccf_ballast: float = 7000,
        lowpass_filter_width: int = 1,
        upsample_filter_width: int = 5,
        max_frames_latency: int = 0,
        frames_per_chunk: int = 0,
        simulate_first_pass_online: bool = False,
        recompute_frame: int = 500,
        nccf_ballast_online: bool = False,
        snip_edges: bool = True,
):
    """Equivalent of `compute-kaldi-pitch-feats`"""
    return torch.ops.tkaldi.ComputeKaldiPitch(
        wave, sample_frequency, frame_length, frame_shift, preemph_coeff,
        min_f0, max_f0, soft_min_f0, penalty_factor, lowpass_cutoff,
        resample_frequency, delta_pitch, nccf_ballast,
        lowpass_filter_width, upsample_filter_width, max_frames_latency,
        frames_per_chunk, simulate_first_pass_online, recompute_frame,
        nccf_ballast_online, snip_edges,
    )
