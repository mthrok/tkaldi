def _init_extension():
    import importlib
    import torch

    module = importlib.util.find_spec('tkaldi._tkaldi')
    if module is None:
        raise ImportError('tkaldi C++ extension module is not available.')
    path = module.origin
    torch.classes.load_library(path)
    torch.ops.load_library(path)


_init_extension()


del _init_extension
