import os
import streamlit as st

# =========================================================
# ⚙️ Evita reload contínuo no Streamlit Cloud
# =========================================================
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"


# =========================================================
# 📦 Imports principais
# =========================================================
import pandas as pd
import numpy as np
import plotly.express as px
from pycaret.classification import load_model


# =========================================================
# 🧭 Configuração da página
# =========================================================
st.set_page_config(page_title="Score de Crédito", page_icon="💳", layout="wide")
st.title("💳 Aplicativo de Escoragem de Crédito")
st.caption("Use este app para escorar novas bases com o modelo treinado (`model_final.pkl`).")


# =========================================================
# 📥 Carregar modelo PyCaret
# =========================================================
@st.cache_resource(show_spinner="🔁 Carregando modelo treinado...")
def carregar_modelo():
    modelo = load_model("model_final")  # sem extensão .pkl
    return modelo


modelo = carregar_modelo()


# =========================================================
# 📂 Upload do CSV
# =========================================================
st.sidebar.header("📂 Upload de Base")
arquivo = st.sidebar.file_uploader("Envie um arquivo CSV", type=["csv"])


# =========================================================
# 🧹 Função auxiliar — preparar base
# =========================================================
def preparar_dados(df: pd.DataFrame, modelo):
    df = df.copy()

    # 1. Remove target, se existir
    for col in ["mau", "target", "y", "classe"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # 2. Converte data_ref para datetime
    if "data_ref" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["data_ref"]):
        try:
            df["data_ref"] = pd.to_datetime(df["data_ref"], errors="coerce")
        except Exception:
            pass

    # 3. Converte categorias para string
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string").str.strip()

    # 4. Alinha colunas esperadas pelo modelo
    if hasattr(modelo, "feature_names_in_"):
        esperadas = list(modelo.feature_names_in_)
        faltantes = [c for c in esperadas if c not in df.columns]
        for c in faltantes:
            df[c] = np.nan
        df = df[esperadas]

    return df


# =========================================================
# 🚀 Escoragem principal
# =========================================================
if arquivo is not None:
    df_raw = pd.read_csv(arquivo)
    st.subheader("🧾 Amostra da base carregada:")
    st.dataframe(df_raw.head())

    with st.spinner("⚙️ Escorando a base..."):
        df_proc = preparar_dados(df_raw, modelo)

        # Faz previsão
        if hasattr(modelo, "predict_proba"):
            score = modelo.predict_proba(df_proc)[:, 1]
        else:
            score = modelo.predict(df_proc)
            try:
                score = score.astype(float)
            except Exception:
                pass

        resultados = df_raw.copy()
        resultados["score"] = score
        resultados["classificacao"] = np.where(resultados["score"] >= 0.5, "Aprovado", "Reprovado")

    st.success("✅ Escoragem concluída!")

    # =====================================================
    # 📊 Métricas resumo
    # =====================================================
    media_score = resultados["score"].mean()
    pct_aprov = (resultados["classificacao"] == "Aprovado").mean() * 100
    pct_reprov = (resultados["classificacao"] == "Reprovado").mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Score Médio", f"{media_score:.2%}")
    col2.metric("Aprovados", f"{pct_aprov:.1f}%")
    col3.metric("Reprovados", f"{pct_reprov:.1f}%")

    # =====================================================
    # 📈 Gráfico 1 — Distribuição dos Scores
    # =====================================================
    st.markdown("### 📊 Distribuição dos Scores")

    fig_hist = px.histogram(
        resultados,
        x="score",
        nbins=30,
        title="Distribuição das Probabilidades de Inadimplência",
        labels={"score": "Score (probabilidade de inadimplência)", "count": "Número de clientes"},
        color_discrete_sequence=["#00B4D8"],
    )
    fig_hist.update_layout(template="plotly_dark", bargap=0.1)
    st.plotly_chart(fig_hist, use_container_width=True)

    # =====================================================
    # ⚖️ Gráfico 2 — Proporção de Aprovação × Reprovação
    # =====================================================
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

    # =====================================================
    # 💾 Botão para download dos resultados
    # =====================================================
    csv_out = resultados.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 Baixar resultados (CSV)",
        data=csv_out,
        file_name="scores_resultados.csv",
        mime="text/csv",
    )

else:
    st.info("Envie um arquivo CSV para iniciar a escoragem.")


