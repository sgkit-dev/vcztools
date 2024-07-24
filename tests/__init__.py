import pytest

# rewrite asserts in assert_vcfs_close to give better failure messages
pytest.register_assert_rewrite("tests.utils")
