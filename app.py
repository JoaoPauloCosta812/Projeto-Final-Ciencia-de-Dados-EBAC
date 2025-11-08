# =====================================================
# Streamlit - Escoragem de base de crédito
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------------------------------
# Configurações da página
# -----------------------------------------------------
st.set_page_config(page_title="Score de Crédito", page_icon="💳", layout="wide")
st.title("💳 Aplicativo de Escoragem de Crédito")
st.caption("Utilize este app para escorar novas bases com o modelo treinado (`model_final.pkl`).")

# -----------------------------------------------------
# Funções auxiliares
# -----------------------------------------------------
@st.cache_resource
def carregar_modelo(caminho_modelo: str):
    """Carrega o modelo treinado."""
    modelo = joblib.load(caminho_modelo)
    return modelo

@st.cache_data
def carregar_csv(arquivo):
    """Carrega arquivo CSV enviado pelo usuário."""
    df = pd.read_csv(arquivo)
    return df

def preprocessar_dados(df: pd.DataFrame):
    """Pipeline simples de pré-processamento."""
    # Exemplo: separa colunas numéricas e categóricas
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    num_tf = Pipeline(steps=[("scaler", StandardScaler())])
    cat_tf = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    pre = ColumnTransformer(
        transformers=[("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)],
        remainder="drop"
    )

    X_prep = pre.fit_transform(df)
    return X_prep, pre

# -----------------------------------------------------
# Upload de arquivo
# -----------------------------------------------------
st.sidebar.header("📂 Upload de Base")
arquivo_csv = st.sidebar.file_uploader("Envie um arquivo CSV", type=["csv"])

modelo_path = "model_final.pkl"  # caminho do modelo treinado
modelo = carregar_modelo(modelo_path)

if arquivo_csv is not None:
    df = carregar_csv(arquivo_csv)
    st.write("### 🧾 Amostra da base carregada:")
    st.dataframe(df.head())

    # Pré-processar e escorar
    with st.spinner("⚙️ Processando e escorando a base..."):
        X_prep, _ = preprocessar_dados(df)
        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(X_prep)[:, 1]
        else:
            proba = modelo.predict(X_prep)

        resultados = df.copy()
        resultados["score"] = proba
        resultados["classificacao"] = np.where(resultados["score"] >= 0.5, "Aprovado", "Reprovado")

    st.success("✅ Escoragem concluída!")
    st.write("### 🔍 Amostra das previsões:")
    st.dataframe(resultados.head())

    # Baixar resultados
    csv = resultados.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 Baixar resultados (CSV)",
        data=csv,
        file_name="scores_resultados.csv",
        mime="text/csv"
    )

else:
    st.info("Envie um arquivo CSV para iniciar a escoragem.")
