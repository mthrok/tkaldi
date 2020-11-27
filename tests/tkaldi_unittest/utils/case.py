import os
from tempfile import TemporaryDirectory

from torch.testing._internal.common_utils import TestCase as _TestCase


class TestCase(_TestCase):
    _temp_dir = None

    @classmethod
    def get_base_temp_dir(cls):
        key = 'TEST_TEMP_DIR'
        if key in os.environ:
            return os.environ[key]
        if cls._temp_dir is None:
            cls._temp_dir = TemporaryDirectory()
        return cls._temp_dir.name

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if cls._temp_dir is not None:
            cls._temp_dir.cleanup()
            cls._temp_dir = None

    def get_temp_path(self, *paths):
        path = os.path.join(self.get_base_temp_dir(), self.id(), *paths)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
