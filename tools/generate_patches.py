#!/usr/bin/env python3

import common


def _main():
    print('Generating patch {common.PATCH_FILE}')
    orig_dir = common.ORIG_DIR.relative_to(common.BASE_DIR)
    work_dir = common.WORK_DIR.relative_to(common.BASE_DIR)

    cmd = ['diff', '-ru', orig_dir, work_dir]
    with open(common.PATCH_FILE, 'w') as file:
        common.call(cmd, cwd=common.BASE_DIR, stdout=file)


if __name__ == '__main__':
    _main()
