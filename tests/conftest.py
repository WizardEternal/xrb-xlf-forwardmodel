"""pytest configuration: make ``src/`` importable and provide shared fixtures."""

import os
import sys

import numpy as np
import pytest
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def pytest_configure(config):
    """Register custom markers (the Phase-2 inference tests run samplers)."""
    config.addinivalue_line(
        "markers", "slow: marks a test that runs a sampler (slower)."
    )


@pytest.fixture(scope="session")
def config():
    """Parsed default config dict."""
    with open(os.path.join(_REPO, "configs", "xlf_defaults.yaml")) as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def rng():
    """A fresh, fixed-seed generator for deterministic tests."""
    return np.random.default_rng(12345)
