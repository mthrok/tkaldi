from subprocess import Popen, PIPE

import torch
import kaldi_io


def run_command_ark(command, waveform):
    """"Run Kaldi command, which expects ark input, on the given Tensor

    Args:
        command (list of str): list of str
        waveform (torch.Tensor): torch.Tensor

    Returns:
        torch.Tensor
    """
    key = 'foo'
    process = Popen(command, stdin=PIPE, stdout=PIPE)
    kaldi_io.write_mat(process.stdin, waveform.cpu().numpy(), key=key)
    process.stdin.close()
    if process.returncode:
        raise RuntimeError('Failed to complete the Kaldi command.')
    result = dict(kaldi_io.read_mat_ark(process.stdout))[key]
    return torch.from_numpy(result.copy())


def run_command_scp(command, path):
    """"Run Klaid command, which expects scp input, on the given Tensor

    Args:
        command (list of str): list of str
        path (str) : Path to the audio file

    Returns:
        torch.Tensor
    """
    key = 'foo'
    process = Popen(command, stdin=PIPE, stdout=PIPE)
    process.stdin.write(f'{key} {path}'.encode('utf8'))
    process.stdin.close()
    if process.returncode:
        raise RuntimeError('Failed to complete the Kaldi command.')
    result = dict(kaldi_io.read_mat_ark(process.stdout))[key]
    return torch.from_numpy(result.copy())


def convert_args(**kwargs):
    args = []
    for key, value in kwargs.items():
        key = '--' + key.replace('_', '-')
        value = str(value)
        if value in ['True', 'False']:
            value = value.lower()
        args.append(f'{key}={value}')
    return args
