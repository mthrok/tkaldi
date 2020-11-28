import os
import subprocess
from pathlib import Path

import torch
import setuptools
from setuptools.command.build_ext import build_ext

_ROOT_DIR = Path(__file__).parent.resolve()
_CSRC_DIR = _ROOT_DIR / 'src' / 'libtkaldi'


def _get_cxx11_abi():
    try:
        return int(torch._C._GLIBCXX_USE_CXX11_ABI)
    except ImportError:
        return 0


class BuildExtension(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-D_GLIBCXX_USE_CXX11_ABI={_get_cxx11_abi()}",
        ]
        build_args = []

        # default to Ninja
        if 'CMAKE_GENERATOR' not in os.environ:
            cmake_args += ["-GNinja"]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", str(_ROOT_DIR)] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

    def get_ext_filename(self, fullname):
        ext_filename = super().get_ext_filename(fullname)
        ext_filename_parts = ext_filename.split('.')
        without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
        ext_filename = '.'.join(without_abi)
        return ext_filename


def _get_ext_modules():
    return [
        setuptools.Extension(
            name='tkaldi.libtkaldi',
            sources=[],
        ),
    ]


def _main():
    setuptools.setup(
        name="tkaldi",
        version="0.0.1",
        description="Experimental project to compile Kaldi with Torch.",
        ext_modules=_get_ext_modules(),
        cmdclass={'build_ext': BuildExtension},
        packages=setuptools.find_packages(where='src'),
        package_dir={'': 'src'},
        install_requires=[
            'torch >= 1.7',
        ],
        zip_safe=False,
    )


if __name__ == '__main__':
    _main()
