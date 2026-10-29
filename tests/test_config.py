import random
from pathlib import Path

import numpy as np
import pytest

from ecbias.config import load_config
from ecbias.utils.seed import set_seed


def test_valid_config_loads():
    config = load_config("configs/default.yaml")
    assert config.seed == 42
    assert config.stats.alpha == 0.05
    assert config.argmining.labels[0] == "CLAIM"


def test_missing_key_raises(tmp_path: Path):
    config_path = tmp_path / "missing.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required config key: paths"):
        load_config(config_path)


def test_seed_propagates():
    set_seed(123)
    first = (random.random(), np.random.random())
    set_seed(123)
    second = (random.random(), np.random.random())
    assert first == second
