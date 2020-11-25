#!/usr/bin/env python3
"""Initialize the workspace / generate the patch / build"""
import argparse
import subprocess
import shutil

from pathlib import Path


_ROOT_DIR = Path(__file__).parent.resolve()
_PATCH_FILE = _ROOT_DIR / 'kaldi.patch'
_KALDI_DIR = _ROOT_DIR / 'third_party' / 'kaldi'
_TKALDI_SRC_DIR = _ROOT_DIR / 'src' / 'libtkaldi'


def _call(commands, cwd=_ROOT_DIR, **kwargs):
    subprocess.check_call(commands, cwd=cwd, **kwargs)


def _init_workspace():
    print('Initializing submodule...')
    _call(['git', 'submodule', 'sync'])
    _call(['git', 'submodule', 'update', '--init', '--recursive'])
    print('Reverting Kaldi ...')
    _call(['git', 'checkout', '.'], cwd=_KALDI_DIR)
    print('Applying patch ...')
    _call(['git', 'apply', str(_PATCH_FILE)], cwd=_KALDI_DIR)


def _generate_patch():
    print('Generating the patch at', _PATCH_FILE.relative_to(_ROOT_DIR))
    with open(_PATCH_FILE, 'w') as file_:
        command = ['git', 'diff']
        _call(command, cwd=_KALDI_DIR, stdout=file_)


def _copy_source_files():
    _sources = [
        'base/io-funcs.h',
        'base/io-funcs-inl.h',
        'base/kaldi-common.h',
        'base/kaldi-error.h',
        'base/kaldi-types.h',
        'base/kaldi-utils.h',
        'base/kaldi-math.h',
        'base/timer.h',
        'matrix/compressed-matrix.h',
        'matrix/matrix-common.h',
        'matrix/matrix-functions.h',
        'matrix/matrix-functions-inl.h',
        'matrix/matrix-lib.h',
        'matrix/optimization.h',
        'matrix/srfft.h',
        'feat/resample.h',
        'feat/resample.cc',
    ]
    for p in _sources:
        src = _KALDI_DIR / 'src' / p
        tgt = _TKALDI_SRC_DIR / 'src' / p
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tgt)


def _build():
    _copy_source_files()
    _call(['pip', 'install', '-e', '.'])


def _diff():
    _call(['git', 'diff'], cwd=_KALDI_DIR)


def _parse_args(subcommands):
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument('subcommand', choices=subcommands)
    return parser.parse_args()


def _main():
    subcommands = {
        'init': _init_workspace,
        'gen': _generate_patch,
        'generate_patch': _generate_patch,
        'build': _build,
        'diff': _diff,
    }
    args = _parse_args(subcommands.keys())
    subcommands[args.subcommand]()
    print('Complete.')


if __name__ == '__main__':
    _main()
