from pathlib import Path
import subprocess

ROOT_DIR = Path(__file__).parent.parent
BASE_DIR = ROOT_DIR / 'workspace'
ORIG_DIR = BASE_DIR / 'orig'
WORK_DIR = BASE_DIR / 'mod'
PATCH_FILE = ROOT_DIR / 'patch' / 'kaldi.patch'


def call(commands, cwd=ROOT_DIR, **kwargs):
    subprocess.check_call(commands, cwd=cwd, **kwargs)
