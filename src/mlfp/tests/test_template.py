# tests/test_example.py
from __future__ import annotations

import pytest

from mlfp import __version__


def test_template() -> None:
    print("Package version is ", __version__)
    assert 0 == 0


if __name__ == "__main__":
    pytest.main()
