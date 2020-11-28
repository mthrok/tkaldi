import os
import shutil
import subprocess
from pathlib import Path

import torch
import setuptools
from setuptools.command.build_ext import build_ext

_ROOT_DIR = Path(__file__).parent.resolve()
_BIN_DIR = _ROOT_DIR / 'src' / 'tkaldi' / 'bin'


class BuildExtension(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        bindir = Path(os.path.join(extdir, 'bin'))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={bindir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
        ]
        build_args = [
            "--verbose",
        ]

        # default to Ninja
        if 'CMAKE_GENERATOR' not in os.environ:
            cmake_args += ["-GNinja"]

        if 'CMAKE_CXX_FLAGS' in os.environ:
            cmake_args += [f"-DCMAKE_CXX_FLAGS={os.environ['CMAKE_CXX_FLAGS']}"]

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

        # Copy binary
        _BIN_DIR.mkdir(parents=True, exist_ok=True)
        for bin_name in ['compute-kaldi-pitch-feats']:
            print(f'Copying {bin_name}')
            shutil.copy2(bindir / bin_name, _BIN_DIR / bin_name)

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
        data_files=[
            ('src/tkaldi/bin', ['compute-kaldi-pitch-feats']),
        ],
        install_requires=[
            'torch >= 1.7',
        ],
        zip_safe=False,
    )


if __name__ == '__main__':
    _main()
