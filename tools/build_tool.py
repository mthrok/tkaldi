#!/usr/bin/env python3
"""Initialize the workspace / generate the patch / build"""
import argparse
import subprocess

from pathlib import Path


_ROOT_DIR = Path(__file__).parent.parent
_PATCH_FILE = _ROOT_DIR / 'patch' / 'kaldi.patch'
_KALDI_DIR = _ROOT_DIR / 'third_party' / 'kaldi'
_KALDI_SRC_DIR = _KALDI_DIR / 'src'
_SOURCE_FILES = [
    Path(p) for p in [
        'feat/resample.h',
        'feat/resample.cc',
    ]
]


def _call(commands, cwd=_ROOT_DIR, **kwargs):
    subprocess.check_call(commands, cwd=cwd, **kwargs)


def _init_workspace():
    print('Initializing submodule...')
    _call(['git', 'submodule', 'sync'])
    _call(['git', 'submodule', 'update', '--init', '--recursive'])
    print('Applying patch ...')
    _call(['git', 'apply', str(_PATCH_FILE)])


def _generate_patch():
    print('Generating the patch...')
    with open(_PATCH_FILE, 'w') as file_:
        command = ['git', 'diff', '--submodule=diff', '--', str(_KALDI_DIR)]
        _call(command, stdout=file_)


def _parse_args(subcommands):
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument('subcommand', choices=subcommands)
    return parser.parse_args()


def _main():
    subcommands = {
        'init': _init_workspace,
        'generate_patch': _generate_patch,
    }
    args = _parse_args(subcommands.keys())
    subcommands[args.subcommand]()


if __name__ == '__main__':
    _main()
