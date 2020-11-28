"""Initialize tkaldi submodules and TorchScript extension"""
from . import (  # noqa: F401 # pylint: disable=unused-import
    feats,
)


def _init_extension():
    # pylint: disable=import-outside-toplevel
    import importlib
    import torch

    module = importlib.util.find_spec('tkaldi.libtkaldi')
    if module is None:
        raise ImportError('tkaldi C++ extension module is not available.')
    path = module.origin
    torch.classes.load_library(path)
    torch.ops.load_library(path)


_init_extension()


del _init_extension
