"""
Módulo de splitting de dados para treino/teste/validação.

Implementa split estratificado por classe (preserva proporção de fraudes)
e validação cruzada com StratifiedKFold.

Decisões metodológicas:
- Split aleatório estratificado (dataset cobre apenas 48h, sem drift relevante)
- Estratificação por Class é OBRIGATÓRIA dado o desbalanceamento extremo
- Random state fixo (42) para reprodutibilidade
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20  # 20% para teste final


def split_train_test(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide o dataset em treino e teste com estratificação.

    A estratificação garante que a proporção de fraudes seja a MESMA
    em treino e teste, o que é crítico para problemas desbalanceados.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame completo com features e target.
    target_col : str
        Nome da coluna alvo. Default: "Class".
    test_size : float
        Proporção do conjunto de teste. Default: 0.20.
    random_state : int
        Semente aleatória para reprodutibilidade.

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Features de treino e teste.
    y_train, y_test : pd.Series
        Target de treino e teste.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,            # ← ESSENCIAL para preservar proporção de fraudes
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test


def get_cv_splitter(
    n_splits: int = 5, random_state: int = RANDOM_STATE
) -> StratifiedKFold:
    """
    Retorna um splitter de validação cruzada estratificado.

    StratifiedKFold preserva a proporção de classes em cada fold —
    obrigatório para datasets desbalanceados.

    Parameters
    ----------
    n_splits : int
        Número de folds. Default: 5.
    random_state : int
        Semente aleatória para reprodutibilidade.

    Returns
    -------
    StratifiedKFold
        Splitter pronto para uso com cross_val_score, GridSearchCV, etc.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def validate_split_balance(
    y_train: pd.Series, y_test: pd.Series, tolerance_pct: float = 0.01
) -> dict:
    """
    Valida que a estratificação preservou a proporção de classes.

    Parameters
    ----------
    y_train, y_test : pd.Series
        Targets de treino e teste.
    tolerance_pct : float
        Diferença máxima aceitável (em pontos percentuais).

    Returns
    -------
    dict
        Métricas de balanceamento.

    Raises
    ------
    ValueError
        Se a diferença exceder a tolerância.
    """
    train_pct = y_train.mean() * 100
    test_pct = y_test.mean() * 100
    diff = abs(train_pct - test_pct)

    result = {
        "train_fraud_pct": round(train_pct, 4),
        "test_fraud_pct": round(test_pct, 4),
        "diff_pct_points": round(diff, 4),
        "within_tolerance": diff <= tolerance_pct,
    }

    if not result["within_tolerance"]:
        raise ValueError(
            f"Estratificação falhou: diferença de {diff:.4f}pp "
            f"excede tolerância de {tolerance_pct}pp"
        )

    return result