"""
Testes unitários para o módulo de feature engineering.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.build_features import (
    add_time_features,
    add_amount_features,
    build_features,
    get_feature_groups,
)


@pytest.fixture
def sample_df():
    """Cria um DataFrame mínimo para testes."""
    return pd.DataFrame({
        "Time": [0, 3600, 7200, 86400, 172800],   # 0h, 1h, 2h, 24h, 48h
        "Amount": [0.0, 10.0, 100.0, 1000.0, 0.0],
        "Class": [0, 0, 1, 0, 1],
    })


def test_add_time_features_creates_columns(sample_df):
    """Time features devem criar Hour, Hour_sin, Hour_cos."""
    result = add_time_features(sample_df)
    assert "Hour" in result.columns
    assert "Hour_sin" in result.columns
    assert "Hour_cos" in result.columns


def test_hour_is_within_24h_range(sample_df):
    """Hour deve estar sempre entre 0 e 24."""
    result = add_time_features(sample_df)
    assert result["Hour"].min() >= 0
    assert result["Hour"].max() < 24


def test_hour_cyclic_encoding_unit_circle(sample_df):
    """sin² + cos² deve ser sempre 1 (propriedade do círculo unitário)."""
    result = add_time_features(sample_df)
    sum_squares = result["Hour_sin"] ** 2 + result["Hour_cos"] ** 2
    np.testing.assert_array_almost_equal(sum_squares.values, np.ones(len(result)))


def test_add_amount_features_creates_columns(sample_df):
    """Amount features devem criar Amount_log, is_amount_zero."""
    result = add_amount_features(sample_df)
    assert "Amount_log" in result.columns
    assert "is_amount_zero" in result.columns


def test_is_amount_zero_flag_correct(sample_df):
    """is_amount_zero deve ser 1 quando Amount=0, 0 caso contrário."""
    result = add_amount_features(sample_df)
    expected = [1, 0, 0, 0, 1]
    assert result["is_amount_zero"].tolist() == expected


def test_amount_log_is_log1p(sample_df):
    """Amount_log deve ser exatamente log1p(Amount)."""
    result = add_amount_features(sample_df)
    np.testing.assert_array_almost_equal(
        result["Amount_log"].values, np.log1p(sample_df["Amount"].values)
    )


def test_build_features_does_not_modify_original(sample_df):
    """build_features não deve modificar o DataFrame original (imutabilidade)."""
    original = sample_df.copy()
    _ = build_features(sample_df)
    pd.testing.assert_frame_equal(sample_df, original)


def test_feature_groups_structure():
    """get_feature_groups deve retornar dict com chaves esperadas."""
    groups = get_feature_groups()
    expected_keys = {"v_features", "amount_features", "time_features", "binary_features", "target"}
    assert set(groups.keys()) == expected_keys
    assert len(groups["v_features"]) == 28