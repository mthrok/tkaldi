#!/usr/bin/env python3
"""Initialize the workspace / generate the patch / build"""
import argparse
import subprocess
import shutil

from pathlib import Path


_ROOT_DIR = Path(__file__).parent
_PATCH_FILE = _ROOT_DIR / 'kaldi.patch'
_KALDI_DIR = _ROOT_DIR / 'third_party' / 'kaldi'
_TKALDI_SRC_DIR = _ROOT_DIR / 'src' / 'libtkaldi'


def _call(commands, cwd=_ROOT_DIR, **kwargs):
    subprocess.check_call(commands, cwd=cwd, **kwargs)


def _init_workspace(_args):
    print('Initializing submodule...')
    _call(['git', 'submodule', 'sync'])
    _call(['git', 'submodule', 'update', '--init', '--recursive'])
    print('Reverting Kaldi ...')
    _call(['git', 'checkout', '.'], cwd=_KALDI_DIR)
    print('Applying patch ...')
    _call(['git', 'apply', str(_PATCH_FILE.resolve())], cwd=_KALDI_DIR)


def _generate_patch(_args):
    print('Generating the patch at', _PATCH_FILE)
    with open(_PATCH_FILE, 'w') as file_:
        command = ['git', 'diff']
        _call(command, cwd=_KALDI_DIR, stdout=file_)


def _clean_src():
    _call(['git', 'clean', '-xdf', 'src'])


def _generate_version_file():
    cwd = _KALDI_DIR / 'src' / 'base'
    _call(['bash', 'get_version.sh'], cwd=cwd)


def _copy_source_files():
    _sources = [
        'base/io-funcs.h',
        'base/io-funcs-inl.h',
        'base/kaldi-common.h',
        'base/kaldi-error.h',
        'base/kaldi-error.cc',
        'base/kaldi-types.h',
        'base/kaldi-utils.h',
        'base/kaldi-math.h',
        'base/timer.h',
        'base/version.h',
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
        print(f'Copying {tgt}')
        shutil.copy2(src, tgt)


def _develop(_args):
    _clean_src()
    _generate_version_file()
    _copy_source_files()
    _call(['pip', 'install', '-e', '.'])


def _diff(args):
    _call(['git', 'diff'] + args, cwd=_KALDI_DIR)


def _parse_args(subcommands):
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument('subcommand', choices=subcommands)
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    return parser.parse_args()


def _main():
    subcommands = {
        'init': _init_workspace,
        'gen': _generate_patch,
        'generate_patch': _generate_patch,
        'dev': _develop,
        'develop': _develop,
        'diff': _diff,
    }
    args = _parse_args(subcommands.keys())
    subcommands[args.subcommand](args.rest)
    print('Complete.')


if __name__ == '__main__':
    _main()
