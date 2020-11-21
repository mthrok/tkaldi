#!/usr/bin/env python3
"""Initialize the workspace"""
import shutil
from pathlib import Path

import common


_KALDI_SRC_DIR = common.ROOT_DIR / 'third_party' / 'kaldi' / 'src'
_SOURCE_FILES = [
    Path(p) for p in [
        'feat/resample.h',
        'feat/resample.cc',
    ]
]


def _copy_file(src, tgt, overwrite=True):
    if overwrite or not tgt.exists():
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tgt)


def _init_submodules():
    common.call(['git', 'submodule', 'sync'])
    common.call(['git', 'submodule', 'update', '--init', '--recursive'])


def _copy_source_files():
    for path in _SOURCE_FILES:
        src = _KALDI_SRC_DIR / path
        tgt = common.ORIG_DIR / path
        _copy_file(src, tgt, overwrite=True)


def _apply_patch():
    cmd = ['patch', '-p0']
    with open(common.PATCH_FILE, 'r') as file:
        common.call(cmd, cwd=common.BASE_DIR, stdin=file)


def _init_workspace():
    if common.WORK_DIR.exists():
        raise RuntimeError(
            'Work directory exists. '
            f'Please remove "{common.WORK_DIR}"')

    print('Copying the original source files...')
    _copy_source_files()
    print('Applying the patch...')
    _apply_patch()
    print('Moving the pathced source to work directory...')
    shutil.move(common.ORIG_DIR, common.WORK_DIR)
    print('Copying the original source files...')
    _copy_source_files()
    print(f'The workspace is ready at {common.WORK_DIR}')


def _main():
    _init_submodules()
    print('Initializing workspace ...')
    _init_workspace()


if __name__ == '__main__':
    _main()
