import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model
import plotly.express as px

# ------------------------------------------------------------
# Config da página
# ------------------------------------------------------------
st.set_page_config(page_title="Score de Crédito", page_icon="💳", layout="wide")
st.title("💳 Aplicativo de Escoragem de Crédito")
st.caption("Use este app para escorar novas bases com o modelo treinado (`model_final.pkl`).")

# ------------------------------------------------------------
# Carregar modelo (cacheado)
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo treinado...")
def carregar_modelo():
    return load_model("model_final")

modelo = carregar_modelo()

# ------------------------------------------------------------
# Upload da base
# ------------------------------------------------------------
st.sidebar.header("📂 Upload de Base")
arquivo = st.sidebar.file_uploader("Envie um arquivo CSV", type=["csv"])

# Inicializa estado de sessão (para evitar recarregar)
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "dados" not in st.session_state:
    st.session_state.dados = None
if "resultados" not in st.session_state:
    st.session_state.resultados = None

# Lê o arquivo apenas quando um novo upload for feito
if arquivo is not None:
    if arquivo.name != st.session_state.uploaded_name:
        st.session_state.uploaded_name = arquivo.name
        st.session_state.dados = pd.read_csv(arquivo)
        st.session_state.resultados = None  # força nova escoragem
else:
    st.info("Envie um arquivo CSV para iniciar a escoragem.")
    st.stop()

df_raw = st.session_state.dados

# ------------------------------------------------------------
# Preparar dados
# ------------------------------------------------------------
def _prepara_df_para_modelo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["mau", "target", "y", "classe"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if "data_ref" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["data_ref"]):
        df["data_ref"] = pd.to_datetime(df["data_ref"], errors="coerce")

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("string").str.strip()

    if hasattr(modelo, "feature_names_in_"):
        cols_esperadas = list(modelo.feature_names_in_)
        faltantes = [c for c in cols_esperadas if c not in df.columns]
        for c in faltantes:
            df[c] = np.nan
        df = df[cols_esperadas]

    return df

# ------------------------------------------------------------
# Escoragem (cacheada manualmente)
# ------------------------------------------------------------
if st.session_state.resultados is None:
    with st.spinner("⚙️ Escorando base..."):
        df = _prepara_df_para_modelo(df_raw)

        if hasattr(modelo, "predict_proba"):
            score = modelo.predict_proba(df)[:, 1]
        else:
            score = modelo.predict(df).astype(float)

        resultados = df_raw.copy()
        resultados["score"] = score
        resultados["classificacao"] = np.where(resultados["score"] >= 0.5, "Aprovado", "Reprovado")
        st.session_state.resultados = resultados
else:
    resultados = st.session_state.resultados

# ------------------------------------------------------------
# Exibição dos resultados
# ------------------------------------------------------------
st.success("✅ Escoragem concluída!")
st.write("### 🔍 Amostra das previsões:")
st.dataframe(resultados.head())

# ------------------------------------------------------------
# 📊 Métricas resumo
# ------------------------------------------------------------
col1, col2, col3 = st.columns(3)
media_score = resultados["score"].mean()
pct_aprov = (resultados["classificacao"] == "Aprovado").mean() * 100
pct_reprov = (resultados["classificacao"] == "Reprovado").mean() * 100

col1.metric("Score Médio", f"{media_score:.2%}")
col2.metric("Aprovados", f"{pct_aprov:.1f}%")
col3.metric("Reprovados", f"{pct_reprov:.1f}%")

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
# 📥 Download
# ------------------------------------------------------------
csv_out = resultados.to_csv(index=False, encoding="utf-8-sig")
st.download_button(
    label="📥 Baixar resultados (CSV)",
    data=csv_out,
    file_name="scores_resultados.csv",
    mime="text/csv",
)
