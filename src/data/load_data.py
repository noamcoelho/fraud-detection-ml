"""
Módulo de carregamento de dados.

Centraliza a lógica de leitura de dataset bruto, com validação básicas
para garantir integridade 
"""

from pathlib import Path 
import pandas as pd


# Caminhos relativos à raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

def load_raw_data(path: Path = DATA_RAW_PATH) -> pd.DataFrame:
    """Carrega o dataset bruto de deteção de fraude.
         Parameters
    ----------
    path : Path, optional
        Caminho para o arquivo CSV. Default: data/raw/creditcard.csv

    Returns
    -------
    pd.DataFrame
        DataFrame com 284807 linhas e 31 colunas.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir no caminho especificado.
    ValueError
        Se o dataset não tiver as dimensões esperadas.    
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {path}. "
            f"Baixe de https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            f"e coloque em data/raw/"
        )

    df = pd.read_csv(path)

    # Validação básica
    expected_shape = (284807, 31)
    if df.shape != expected_shape:
        raise ValueError(
            f"Shape inesperado: {df.shape}. Esperado: {expected_shape}"
        )

    if "Class" not in df.columns:
        raise ValueError("Coluna 'Class' (target) ausente no dataset")

    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Retorna informações básicas sobre o dataset.

    Útil para sanity checks rápidos em notebooks ou pipelines.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser inspecionado.

    Returns
    -------
    dict
        Dicionário com shape, missing values, distribuição de classes, etc.
    """
    n_fraud = int((df["Class"] == 1).sum())
    n_legit = int((df["Class"] == 0).sum())

    return {
        "shape": df.shape,
        "n_features": df.shape[1] - 1,
        "n_samples": df.shape[0],
        "missing_values": int(df.isnull().sum().sum()),
        "n_fraud": n_fraud,
        "n_legit": n_legit,
        "fraud_rate_pct": round(n_fraud / len(df) * 100, 4),
        "imbalance_ratio": round(n_legit / n_fraud, 1),
    }