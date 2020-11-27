from typing import Union

import scipy
import torch


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize the input tensor values to [-1, 1]"""
    if tensor.dtype == torch.float32:
        pass
    elif tensor.dtype == torch.int32:
        tensor = tensor.to(torch.float32)
        tensor[tensor > 0] /= 2147483647.
        tensor[tensor < 0] /= 2147483648.
    elif tensor.dtype == torch.int16:
        tensor = tensor.to(torch.float32)
        tensor[tensor > 0] /= 32767.
        tensor[tensor < 0] /= 32768.
    elif tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32) - 128
        tensor[tensor > 0] /= 127.
        tensor[tensor < 0] /= 128.
    return tensor


def unnormalize(tensor: torch.tensor, dtype: torch.dtype):
    """Convert input tensor with values between -1 and 1 to integer encoding

    Args:
        tensor: input tensor, assumed between -1 and 1
        dtype: desired output tensor dtype
    Returns:
        Tensor: shape of (n_channels, sample_rate * duration)
    """
    if dtype == torch.int32:
        tensor *= (tensor > 0) * 2147483647 + (tensor < 0) * 2147483648
    if dtype == torch.int16:
        tensor *= (tensor > 0) * 32767 + (tensor < 0) * 32768
    if dtype == torch.uint8:
        tensor *= (tensor > 0) * 127 + (tensor < 0) * 128
        tensor += 128
    tensor = tensor.to(dtype)
    return tensor


def get_sinusoid(
        *,
        sample_rate: int,
        dtype: Union[str, torch.dtype],
        frequency: float = 300,
        duration: float = 1,  # seconds
        num_channels: int = 1,
        device: Union[str, torch.device] = "cpu",
        channels_first: bool = True,
):
    """Generate pseudo audio data with sine wave.

    Args:
        frequency: Frequency of sine wave
        sample_rate: Sampling rate
        duration: Length of the resulting Tensor in seconds.
        num_channels: Number of channels
        dtype: Torch dtype
        device: device

    Returns:
        Tensor: shape of (num_channels, int(sample_rate * duration))
    """
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    pie2 = 2 * 3.141592653589793
    end = pie2 * frequency * duration
    num_samples = int(sample_rate * duration)
    theta = torch.linspace(
        0, end, num_samples, dtype=torch.float32, device=device)
    tensor = torch.sin(theta).repeat([num_channels, 1])
    if not channels_first:
        tensor = tensor.t()
    return unnormalize(tensor, dtype)
