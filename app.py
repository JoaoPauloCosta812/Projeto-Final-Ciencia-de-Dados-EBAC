import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model
import plotly.express as px


# Config da página
st.set_page_config(page_title="Score de Crédito", page_icon="💳", layout="wide")
st.title("💳 Aplicativo de Escoragem de Crédito")
st.caption("Use este app para escorar novas bases com o modelo treinado (`model_final.pkl`).")


# Carregar modelo (PyCaret)
@st.cache_resource
def carregar_modelo():
    modelo = load_model("model_final")  # sem .pkl
    return modelo

modelo = carregar_modelo()


# Upload
st.sidebar.header("📂 Upload de Base")
arquivo = st.sidebar.file_uploader("Envie um arquivo CSV", type=["csv"])

def _prepara_df_para_modelo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Remover target se vier por engano
    for col in ["mau", "target", "y", "classe"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # 2) Converter data_ref
    if "data_ref" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["data_ref"]):
        try:
            df["data_ref"] = pd.to_datetime(df["data_ref"], errors="coerce")
        except Exception:
            pass

    # 3) Tipos categóricos como string
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("string").str.strip()

    # 4) Ajustar colunas conforme o modelo
    if hasattr(modelo, "feature_names_in_"):
        cols_esperadas = list(modelo.feature_names_in_)
        faltantes = [c for c in cols_esperadas if c not in df.columns]
        for c in faltantes:
            df[c] = np.nan
        df = df[cols_esperadas]

    return df


if arquivo is not None:
    df_raw = pd.read_csv(arquivo)
    st.write("### 🧾 Amostra da base carregada:")
    st.dataframe(df_raw.head())

    with st.spinner("⚙️ Processando e escorando a base..."):
        df = _prepara_df_para_modelo(df_raw)

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


    # Métricas resumo
    col1, col2, col3 = st.columns(3)
    media_score = resultados["score"].mean()
    pct_aprov = (resultados["classificacao"] == "Aprovado").mean() * 100
    pct_reprov = (resultados["classificacao"] == "Reprovado").mean() * 100

    col1.metric("Score Médio", f"{media_score:.2%}")
    col2.metric("Aprovados", f"{pct_aprov:.1f}%")
    col3.metric("Reprovados", f"{pct_reprov:.1f}%")


    # Gráfico 1 — Distribuição dos Scores
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


    # ⚖️ Gráfico 2 — Proporção de Aprovação × Reprovação
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


    # Download
    csv_out = resultados.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 Baixar resultados (CSV)",
        data=csv_out,
        file_name="scores_resultados.csv",
        mime="text/csv",
    )

else:
    st.info("Envie um arquivo CSV para iniciar a escoragem.")
