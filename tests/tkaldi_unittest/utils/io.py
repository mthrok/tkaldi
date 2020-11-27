import torch
import scipy.io.wavfile

from .data import normalize as normalize_


def save_wav(path, data, sample_rate):
    """Save wav file"""
    scipy.io.wavfile.write(path, sample_rate, data.numpy())


def load_wav(path: str, normalize=True, channels_first=True) -> torch.Tensor:
    """Load wav file without torchaudio"""
    sample_rate, data = scipy.io.wavfile.read(path)
    data = torch.from_numpy(data.copy())
    if data.ndim == 1:
        data = data.unsqueeze(1)
    if normalize:
        data = normalize_(data)
    if channels_first:
        data = data.transpose(1, 0)
    return data, sample_rate
