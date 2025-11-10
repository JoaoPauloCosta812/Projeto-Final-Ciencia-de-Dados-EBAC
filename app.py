import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model

# ------------------------------------------------------------
# Config da página
# ------------------------------------------------------------
st.set_page_config(page_title="Score de Crédito", page_icon="💳", layout="wide")
st.title("💳 Aplicativo de Escoragem de Crédito")
st.caption("Use este app para escorar novas bases com o modelo treinado (`model_final.pkl`).")

# ------------------------------------------------------------
# Carregar modelo (PyCaret)
#   -> use o nome SEM extensão se o arquivo se chama model_final.pkl
#      e está na raiz do projeto.
# ------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    # Se o arquivo é 'model_final.pkl', use load_model('model_final')
    modelo = load_model("model_final")
    return modelo

modelo = carregar_modelo()

# ------------------------------------------------------------
# Upload
# ------------------------------------------------------------
st.sidebar.header("📂 Upload de Base")
arquivo = st.sidebar.file_uploader("Envie um arquivo CSV", type=["csv"])

def _prepara_df_para_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """Ajustes mínimos para garantir compatibilidade com o pipeline salvo."""
    df = df.copy()

    # 1) Remover target se vier por engano
    for col in ["mau", "target", "y", "classe"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # 2) Garantir que a coluna de data seja datetime (se existir)
    if "data_ref" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["data_ref"]):
        # tenta converter sem quebrar se já estiver ok
        try:
            df["data_ref"] = pd.to_datetime(df["data_ref"], errors="coerce")
        except Exception:
            pass

    # 3) Tipos categóricos como string (evita categorias “object” inconsistentes)
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    for c in obj_cols:
        df[c] = df[c].astype("string").str.strip()

    # 4) Opcional: alinhar colunas esperadas pelo modelo, se disponível
    if hasattr(modelo, "feature_names_in_"):
        cols_esperadas = list(modelo.feature_names_in_)
        # mantém só as que existem
        df = df[[c for c in cols_esperadas if c in df.columns]]
        # se faltar alguma coluna esperada, cria vazia (NaN) — a pipeline tratará
        faltantes = [c for c in cols_esperadas if c not in df.columns]
        for c in faltantes:
            df[c] = np.nan
        # reordena
        df = df[cols_esperadas]

    return df

if arquivo is not None:
    df_raw = pd.read_csv(arquivo)
    st.write("### 🧾 Amostra da base carregada:")
    st.dataframe(df_raw.head())

    with st.spinner("⚙️ Processando e escorando a base..."):
        df = _prepara_df_para_modelo(df_raw)

        # IMPORTANTÍSSIMO: usar diretamente predict / predict_proba do pipeline
        # do PyCaret. NÃO usar predict_model(...).
        if hasattr(modelo, "predict_proba"):
            score = modelo.predict_proba(df)[:, 1]
        else:
            # fallback: alguns modelos não têm proba; usamos a classe
            score = modelo.predict(df)
            # normaliza para [0,1] se veio como {0,1}
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
    csv_out = resultados.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 Baixar resultados (CSV)",
        data=csv_out,
        file_name="scores_resultados.csv",
        mime="text/csv",
    )
else:
    st.info("Envie um arquivo CSV para iniciar a escoragem.")

