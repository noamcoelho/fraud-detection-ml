"""
Testes para o módulo de splitting.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import split_train_test, get_cv_splitter, validate_split_balance


@pytest.fixture
def imbalanced_df():
    """DataFrame imbalanced para simular nosso cenário."""
    np.random.seed(42)
    n = 10000
    n_pos = 50  # 0.5% positivos
    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
        "Class": [1] * n_pos + [0] * (n - n_pos),
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def test_split_preserves_class_distribution(imbalanced_df):
    """Estratificação deve preservar proporção de classes (~)."""
    X_train, X_test, y_train, y_test = split_train_test(imbalanced_df)
    diff = abs(y_train.mean() - y_test.mean())
    assert diff < 0.005, f"Diferença de proporção muito grande: {diff}"


def test_split_sizes_correct(imbalanced_df):
    """test_size=0.2 deve resultar em 80/20."""
    X_train, X_test, _, _ = split_train_test(imbalanced_df, test_size=0.2)
    expected_test_size = int(len(imbalanced_df) * 0.2)
    assert abs(len(X_test) - expected_test_size) <= 1


def test_split_no_overlap(imbalanced_df):
    """Treino e teste não podem ter índices em comum."""
    X_train, X_test, _, _ = split_train_test(imbalanced_df)
    overlap = set(X_train.index) & set(X_test.index)
    assert len(overlap) == 0


def test_split_reproducibility(imbalanced_df):
    """Mesmo random_state deve dar mesmo split."""
    s1 = split_train_test(imbalanced_df, random_state=42)
    s2 = split_train_test(imbalanced_df, random_state=42)
    pd.testing.assert_frame_equal(s1[0], s2[0])


def test_cv_splitter_n_splits():
    """get_cv_splitter deve retornar splitter com n_splits correto."""
    cv = get_cv_splitter(n_splits=5)
    assert cv.n_splits == 5


def test_validate_split_balance_passes(imbalanced_df):
    """validate_split_balance deve passar com split estratificado."""
    _, _, y_train, y_test = split_train_test(imbalanced_df)
    result = validate_split_balance(y_train, y_test)
    assert result["within_tolerance"]


def test_validate_split_balance_fails():
    """validate_split_balance deve falhar com split desequilibrado."""
    y_train = pd.Series([0] * 950 + [1] * 50)        # 5% fraude
    y_test = pd.Series([0] * 990 + [1] * 10)         # 1% fraude
    with pytest.raises(ValueError):
        validate_split_balance(y_train, y_test, tolerance_pct=0.5)