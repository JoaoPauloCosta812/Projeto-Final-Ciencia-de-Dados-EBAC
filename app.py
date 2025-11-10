import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model
import plotly.express as px
from pathlib import Path

# ------------------------------------------------------------
# Configurações da página
# ------------------------------------------------------------
st.set_page_config(page_title="Score de Crédito", page_icon="💳", layout="wide")
st.title("💳 Aplicativo de Escoragem de Crédito")
st.caption("Use este app para escorar novas bases com o modelo treinado (`model_final.pkl`).")

# ------------------------------------------------------------
# Caminhos organizados do projeto
# ------------------------------------------------------------
BASE_PATH = Path("Projeto-Final-Ciencia-de-Dados-EBAC/data/base")
MODELO_PATH = BASE_PATH / "model_final"
DEFAULT_CSV_PATH = BASE_PATH / "credit_scoring_para_streamlit_corrigido.csv"

# ------------------------------------------------------------
# Carregar modelo
# ------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    modelo = load_model(str(MODELO_PATH))
    return modelo

modelo = carregar_modelo()

# ------------------------------------------------------------
# Upload de arquivo
# ------------------------------------------------------------
st.sidebar.header("📂 Upload de Base")
arquivo = st.sidebar.file_uploader("Envie um arquivo CSV", type=["csv"])

def preparar_df_para_modelo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Remover target se existir
    for col in ["mau", "target", "y", "classe"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # 2) Garantir tipo datetime
    if "data_ref" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["data_ref"]):
        try:
            df["data_ref"] = pd.to_datetime(df["data_ref"], errors="coerce")
        except Exception:
            pass

    # 3) Converter strings
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    for c in obj_cols:
        df[c] = df[c].astype("string").str.strip()

    # 4) Ajustar colunas conforme modelo
    if hasattr(modelo, "feature_names_in_"):
        cols_esperadas = list(modelo.feature_names_in_)
        df = df[[c for c in cols_esperadas if c in df.columns]]
        faltantes = [c for c in cols_esperadas if c not in df.columns]
        for c in faltantes:
            df[c] = np.nan
        df = df[cols_esperadas]

    return df

# ------------------------------------------------------------
# Processamento e escoragem
# ------------------------------------------------------------
if arquivo is not None:
    df_raw = pd.read_csv(arquivo)
else:
    st.sidebar.info("📄 Nenhum arquivo enviado — usando base padrão.")
    df_raw = pd.read_csv(DEFAULT_CSV_PATH)

st.write("### 🧾 Amostra da base carregada:")
st.dataframe(df_raw.head())

with st.spinner("⚙️ Processando e escorando a base..."):
    df = preparar_df_para_modelo(df_raw)

    # Escorar direto com o pipeline (sem predict_model)
    if hasattr(modelo, "predict_proba"):
        score = modelo.predict_proba(df)[:, 1]
    else:
        score = modelo.predict(df)
        try:
            score = score.astype(float)
        except Exception:
            pass

    resultados = df_raw.copy()
    resultados["score"] = score
    resultados["classificacao"] = np.where(resultados["score"] >= 0.5, "Aprovado", "Reprovado")

st.success("✅ Escoragem concluída!")
st.write("### 🔍 Amostra das previsões:")
st.dataframe(resultados.head())

# ------------------------------------------------------------
# 📊 Gráfico 1 — Distribuição dos Scores
# ------------------------------------------------------------
st.markdown("### 📊 Distribuição dos Scores")

fig_hist = px.histogram(
    resultados,
    x="score",
    nbins=30,
    title="Distribuição das Probabilidades de Inadimplência",
    labels={"score": "Score (probabilidade de inadimplência)", "count": "Número de clientes"},
    color_discrete_sequence=["#00B4D8"]
)
fig_hist.update_layout(template="plotly_dark", bargap=0.1)
st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------------------------------------------
# ⚖️ Gráfico 2 — Proporção de Aprovação × Reprovação
# ------------------------------------------------------------
st.markdown("### ⚖️ Proporção de Aprovações e Reprovações")

graf_counts = resultados["classificacao"].value_counts(normalize=True).mul(100).reset_index()
graf_counts.columns = ["classificacao", "percentual"]

fig_pie = px.pie(
    graf_counts,
    names="classificacao",
    values="percentual",
    title="Distribuição de Classificação dos Clientes",
    color="classificacao",
    color_discrete_map={"Aprovado": "#00B050", "Reprovado": "#C00000"},
)
fig_pie.update_traces(textinfo="percent+label")
st.plotly_chart(fig_pie, use_container_width=True)

# ------------------------------------------------------------
# 📥 Botão de download
# ------------------------------------------------------------
csv_out = resultados.to_csv(index=False, encoding="utf-8-sig")
st.download_button(
    label="📥 Baixar resultados (CSV)",
    data=csv_out,
    file_name="scores_resultados.csv",
    mime="text/csv",
)
