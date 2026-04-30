#  Detecção de Fraude em Transações de Cartão de Crédito

Projeto de Machine Learning para identificação de transações fraudulentas em tempo real, com foco em desbalanceamento extremo de classes (~0.17% de fraudes).

##  Sobre o Projeto

**Problema:** Identificar transações fraudulentas otimizando o trade-off entre detectar o máximo de fraudes (recall) e minimizar bloqueios indevidos (precision).

**Tipo:** Classificação binária supervisionada.

**Métrica principal:** PR-AUC (Precision-Recall Area Under Curve).

**Metas:**
- PR-AUC ≥ 0.80
- Recall ≥ 0.85
- Precision ≥ 0.70

##  Estrutura do Projeto

\`\`\`
fraud-detection-ml/
├── data/              # Dados (não versionados)
├── notebooks/         # Análises e experimentos
├── src/               # Código modular reutilizável
├── models/            # Modelos treinados (não versionados)
├── reports/           # Relatórios e figuras
├── tests/             # Testes unitários
└── api/               # Aplicação FastAPI (deploy)
\`\`\`

##  Como Reproduzir

### 1. Clone o repositório
\`\`\`bash
git clone https://github.com/noamcoelho/fraud-detection-ml.git
cd fraud-detection-ml
\`\`\`

### 2. Crie o ambiente virtual
\`\`\`bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\Activate.ps1  # Windows
\`\`\`

### 3. Instale as dependências
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Baixe o dataset
- Acesse: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Faça o download e coloque `creditcard.csv` em `data/raw/`

### 5. Execute os notebooks
Abra `notebooks/01_eda.ipynb` no VSCode/Jupyter.

##  Stack Técnica

- **Linguagem:** Python 3.10+
- **ML:** scikit-learn, XGBoost, LightGBM, imbalanced-learn
- **Interpretabilidade:** SHAP
- **Otimização:** Optuna
- **API:** FastAPI
- **Visualização:** Matplotlib, Seaborn, Plotly

##  Status do Projeto

- [x] Etapa 1 — Definição do Problema
- [x] Etapa 2 — Estrutura e Aquisição
- [ ] Etapa 3 — Versionamento Git
- [ ] Etapa 4 — EDA
- [ ] Etapa 5 — Pré-processamento
- [ ] Etapa 6 — Prevenção de Data Leakage
- [ ] Etapa 7 — Modelagem
- [ ] Etapa 8 — Validação
- [ ] Etapa 9 — Interpretação (SHAP)
- [ ] Etapa 10 — Melhorias
- [ ] Etapa 11 — Deploy (FastAPI)
- [ ] Etapa 12 — Comunicação

##  Licença

Este projeto está sob licença MIT. O dataset utilizado segue a licença ODbL.

##  Autor

Noam COelho — coelhonoam@gmail.com — Noam Coelho no LinkedIn