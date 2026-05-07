"""
Módulo de feature engineering.

Cria features derivadas baseadas nas decisões tomadas na EDA:
- Hour: hora do dia (cíclica)
- Hour_sin / Hour_cos: encoding cíclico para preservar circularidade
- Amount_log: log1p(Amount) para reduzir skewness
- is_amount_zero: flag para card testing
"""

import numpy as np     
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features temporais a partir da coluna Time.

    A coluna Time é em segundos desde a primeira transação.
    Convertemos para hora do dia (0-24) e aplicamos encoding cíclico,
    preservando a propriedade de que 23h e 0h estão "próximos".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com coluna 'Time'.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas adicionais: Hour, Hour_sin, Hour_cos.
    """
    df = df.copy()  # evita modificar o original
    df["Hour"] = (df["Time"] / 3600) % 24

    # Encoding cíclico: preserva proximidade entre 23h e 0h
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    return df

def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas de Amount.

    - Amount_log: log1p para reduzir skewness extrema (mediana fraude << média)
    - is_amount_zero: flag binária para detectar card testing
                       (27 fraudes têm Amount=0 segundo EDA)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com coluna 'Amount'.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas adicionais: Amount_log, is_amount_zero.
    """
    df = df.copy() 
    df["Amount_log"] = np.log1p(df["Amount"])
    df["is_amount_zero"] = (df["Amount"] == 0).astype(int)
    
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas as transformações de feature engineering.

    Pipeline completo:
    1. Features temporais (Hour, Hour_sin, Hour_cos)
    2. Features de valor (Amount_log, is_amount_zero)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame bruto com colunas Time, Amount, V1-V28, Class.

    Returns
    -------
    pd.DataFrame
        DataFrame enriquecido com features derivadas.
    """
    df = add_time_features(df)
    df = add_amount_features(df)
    
    return df

def get_feature_groups() -> dict:
    """
    Retorna os grupos de features para uso em pipelines.

    Útil para o ColumnTransformer (Etapa 5.5) saber o que padronizar.

    Returns
    -------
    dict
        Dicionário com listas de features por categoria.
    """
    return {
        "v_features": [f"V{i}" for i in range(1, 29)],
        "amount_features": ["Amount", "Amount_log"],
        "time_features": ["Time", "Hour", "Hour_sin", "Hour_cos"],
        "binary_features": ["is_amount_zero"],
        "target": "Class",
    }