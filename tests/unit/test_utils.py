# Path: tests/unit/test_utils.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi

import numpy as np
import pytest

from ml_unsupervised.utils import as_2d_float_array, coerce_scalar, normalize_name, parse_key_value_params


def test_normalize_and_coerce() -> None:
    assert normalize_name("  K-Means ") == "k_means"
    assert coerce_scalar("true") is True
    assert coerce_scalar("false") is False
    assert coerce_scalar("none") is None
    assert coerce_scalar("3") == 3
    assert coerce_scalar("3.5") == 3.5
    assert coerce_scalar("abc") == "abc"


def test_parse_key_value_params() -> None:
    assert parse_key_value_params(["n_clusters=3", "flag=true", "x=1.5"]) == {
        "n_clusters": 3,
        "flag": True,
        "x": 1.5,
    }
    assert parse_key_value_params(None) == {}
    with pytest.raises(ValueError):
        parse_key_value_params(["bad"])
    with pytest.raises(ValueError):
        parse_key_value_params(["=1"])


def test_as_2d_float_array_validation() -> None:
    arr = as_2d_float_array([[1, 2], [3, 4]])
    assert arr.dtype == float and arr.shape == (2, 2)
    with pytest.raises(ValueError):
        as_2d_float_array([1, 2, 3])
    with pytest.raises(ValueError):
        as_2d_float_array([[]])
    with pytest.raises(ValueError):
        as_2d_float_array([[1.0, np.inf]])
