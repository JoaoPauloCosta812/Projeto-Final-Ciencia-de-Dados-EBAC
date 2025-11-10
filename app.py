import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pycaret.classification import load_model

# =========================================
# Configurações da página
# =========================================
st.set_page_config(page_title="Score de Crédito", page_icon="💳", layout="wide")
st.title("💳 Aplicativo de Escoragem de Crédito")
st.caption("Use este app para escorar novas bases com o modelo treinado (`model_final.pkl`).")

# =========================================
# Controle de estado persistente
# =========================================
if "modelo" not in st.session_state:
    st.session_state.modelo = None
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "resultados" not in st.session_state:
    st.session_state.resultados = None
if "ultimo_arquivo" not in st.session_state:
    st.session_state.ultimo_arquivo = None

# =========================================
# Funções auxiliares
# =========================================
@st.cache_resource(show_spinner="Carregando modelo treinado...")
def carregar_modelo():
    return load_model("model_final")

def preparar_dados(df: pd.DataFrame, modelo):
    df = df.copy()
    for col in ["mau", "target", "y", "classe"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if "data_ref" in df.columns:
        try:
            df["data_ref"] = pd.to_datetime(df["data_ref"], errors="coerce")
        except Exception:
            pass

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("string").str.strip()

    if hasattr(modelo, "feature_names_in_"):
        cols = list(modelo.feature_names_in_)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[cols]
    return df

def escorar(df_raw, modelo):
    df = preparar_dados(df_raw, modelo)
    if hasattr(modelo, "predict_proba"):
        score = modelo.predict_proba(df)[:, 1]
    else:
        score = modelo.predict(df).astype(float)
    out = df_raw.copy()
    out["score"] = score
    out["classificacao"] = np.where(out["score"] >= 0.5, "Aprovado", "Reprovado")
    return out

# =========================================
# Sidebar - Upload
# =========================================
st.sidebar.header("📂 Upload de Base")
arquivo = st.sidebar.file_uploader("Envie um arquivo CSV", type=["csv"])

# =========================================
# Execução principal (lazy loading)
# =========================================
if arquivo is not None:
    if st.session_state.ultimo_arquivo != arquivo.name:
        st.session_state.ultimo_arquivo = arquivo.name
        st.session_state.df_raw = pd.read_csv(arquivo)
        st.session_state.modelo = carregar_modelo()
        st.session_state.resultados = escorar(st.session_state.df_raw, st.session_state.modelo)
        st.toast("✅ Base escorada com sucesso!", icon="✅")

    resultados = st.session_state.resultados
    df_raw = st.session_state.df_raw

    # Exibir preview
    st.write("### 🧾 Amostra da base carregada:")
    st.dataframe(df_raw.head())

    # Exibir resultados
    st.write("### 🔍 Amostra das previsões:")
    st.dataframe(resultados.head())

    # =========================================
    # Métricas resumo
    # =========================================
    col1, col2, col3 = st.columns(3)
    media_score = resultados["score"].mean()
    pct_aprov = (resultados["classificacao"] == "Aprovado").mean() * 100
    pct_reprov = (resultados["classificacao"] == "Reprovado").mean() * 100

    col1.metric("Score Médio", f"{media_score:.2%}")
    col2.metric("Aprovados", f"{pct_aprov:.1f}%")
    col3.metric("Reprovados", f"{pct_reprov:.1f}%")

    # =========================================
    # Gráficos
    # =========================================
    st.markdown("### 📊 Distribuição dos Scores")
    fig_hist = px.histogram(
        resultados, x="score", nbins=30,
        title="Distribuição das Probabilidades de Inadimplência",
        color_discrete_sequence=["#00B4D8"]
    )
    fig_hist.update_layout(template="plotly_dark", bargap=0.1)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### ⚖️ Proporção de Aprovações e Reprovações")
    graf_counts = resultados["classificacao"].value_counts(normalize=True).mul(100).reset_index()
    graf_counts.columns = ["classificacao", "percentual"]
    fig_pie = px.pie(
        graf_counts, names="classificacao", values="percentual",
        title="Distribuição de Classificação dos Clientes",
        color="classificacao",
        color_discrete_map={"Aprovado": "#00B050", "Reprovado": "#C00000"}
    )
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

    # =========================================
    # Download
    # =========================================
    csv_out = resultados.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 Baixar resultados (CSV)",
        data=csv_out,
        file_name="scores_resultados.csv",
        mime="text/csv",
    )
else:
    st.info("Envie um arquivo CSV para iniciar a escoragem.")
