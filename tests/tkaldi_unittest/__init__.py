import os


if 'TORCH_SHOW_CPP_STACKTRACES' not in os.environ:
    os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
