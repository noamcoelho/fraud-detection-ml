"""
Módulo de pré-processamento usando Pipeline scikit-learn.

Encapsula TODAS as transformações que dependem de dados (médias, desvios,
limites) em um Pipeline que é treinado APENAS no fold de treino e
aplicado ao fold de teste — eliminando data leakage por construção.

Estratégia:
- V1-V28: já estão padronizadas pelo PCA original — NÃO tocar
- Amount, Amount_log: padronizar com StandardScaler
- Time, Hour: padronizar
- Hour_sin, Hour_cos: já em [-1, 1] — NÃO tocar
- is_amount_zero: binária — NÃO tocar
"""

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor(
    columns_to_scale: List[str],
    columns_passthrough: List[str],
) -> ColumnTransformer:
    """
    Constrói o ColumnTransformer com a estratégia de pré-processamento.

    O ColumnTransformer aplica StandardScaler apenas nas colunas que
    precisam, mantendo as outras intactas. Ao envolver isso em um
    Pipeline, garantimos que o fit ocorra apenas nos dados de treino.

    Parameters
    ----------
    columns_to_scale : list of str
        Nomes das colunas que devem ser padronizadas (média 0, std 1).
    columns_passthrough : list of str
        Colunas que devem passar sem transformação (já padronizadas
        ou binárias).

    Returns
    -------
    ColumnTransformer
        Transformer pronto para uso em Pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), columns_to_scale),
            ("pass", "passthrough", columns_passthrough),
        ],
        remainder="drop",   # qualquer coluna não listada é descartada
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_preprocessing_columns() -> dict:
    """
    Retorna a configuração de colunas para o preprocessor.

    Esta função encapsula nossas decisões da EDA:
    - V's: passthrough (já padronizadas pelo PCA)
    - Amount/Time: scale (escalas muito diferentes)
    - Encoding cíclico e binárias: passthrough
    - Time, Amount, Hour: removidas (mantemos só as derivadas)

    Returns
    -------
    dict
        {'to_scale': [...], 'passthrough': [...]}
    """
    v_features = [f"V{i}" for i in range(1, 29)]

    columns_to_scale = [
        "Amount_log",   # log já reduziu skew, mas escala continua diferente das V's
        # Amount original removido — Amount_log é uma versão melhor
        # Time removido — usamos Hour_sin/cos no lugar
    ]

    columns_passthrough = (
        v_features                              # 28 features PCA já padronizadas
        + ["Hour_sin", "Hour_cos"]              # encoding cíclico já em [-1, 1]
        + ["is_amount_zero"]                    # flag binária
    )

    return {
        "to_scale": columns_to_scale,
        "passthrough": columns_passthrough,
        "all_used": columns_to_scale + columns_passthrough,
    }


def build_full_pipeline(estimator) -> Pipeline:
    """
    Constrói o pipeline completo: preprocessor + estimator.

    Esta é a função principal que vamos usar na Etapa 7 (modelagem).
    Ao usar Pipeline, garantimos que:
    1. O fit do scaler ocorre apenas no treino de cada fold
    2. O transform é aplicado igualmente ao teste
    3. Não há vazamento estatístico

    Parameters
    ----------
    estimator : sklearn estimator
        Qualquer modelo sklearn-compatible (LogisticRegression, etc).

    Returns
    -------
    Pipeline
        Pipeline com 2 etapas: 'preprocessor' e 'classifier'.
    """
    cols = get_preprocessing_columns()
    preprocessor = build_preprocessor(
        columns_to_scale=cols["to_scale"],
        columns_passthrough=cols["passthrough"],
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])

    return pipeline