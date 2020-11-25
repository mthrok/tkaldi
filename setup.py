import os
import setuptools
from pathlib import Path

from torch.utils.cpp_extension import CppExtension, BuildExtension

_ROOT_DIR = Path(__file__).parent.resolve()
_CSRC_DIR = _ROOT_DIR / 'src' / 'libtkaldi'


def _get_srcs():
    return [str(p) for p in _CSRC_DIR.glob('**/*.cc')]


def _get_include_dirs():
    return [
        str(_CSRC_DIR / 'src'),
    ]


def _get_ext_modules():
    return [
        CppExtension(
            name='tkaldi._tkaldi',
            sources=_get_srcs(),
            include_dirs=_get_include_dirs(),
        ),
    ]


def _main():
    setuptools.setup(
        name="tkaldi",
        version="0.0.1",
        description="Experimental project to compile Kaldi with Torch.",
        ext_modules=_get_ext_modules(),
        cmdclass={
            'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
        },
        packages=setuptools.find_packages(where='src'),
        package_dir={'': 'src'},
        install_requires=[
            'torch >= 1.7',
        ],
        zip_safe=False,
    )


if __name__ == '__main__':
    _main()
