# ==============================================================
# APP: Aplicativo de Escoragem de Crédito
# Autor: João Paulo Costa
# Projeto Final EBAC × Semantix
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model


# ==============================================================
# Configurações da página
# ==============================================================
st.set_page_config(page_title="Score de Crédito", page_icon="💳", layout="wide")
st.title("💳 Aplicativo de Escoragem de Crédito")
st.caption("Utilize este app para escorar novas bases com o modelo treinado (`model_final.pkl`).")


# ==============================================================
# Funções auxiliares
# ==============================================================
@st.cache_resource
def carregar_modelo(caminho_modelo: str):
    """
    Carrega o modelo salvo pelo PyCaret (sem precisar da extensão .pkl).
    """
    modelo = load_model(caminho_modelo.replace(".pkl", ""))  # PyCaret busca o nome base
    return modelo


@st.cache_data
def carregar_csv(arquivo):
    """
    Lê o arquivo CSV enviado pelo usuário.
    """
    return pd.read_csv(arquivo)


# ==============================================================
# Interface principal
# ==============================================================
st.sidebar.header("📂 Upload de Base")
arquivo_csv = st.sidebar.file_uploader("Envie um arquivo CSV", type=["csv"])

# Nome do modelo salvo
modelo_path = "model_final"

# Carrega o modelo
modelo = carregar_modelo(modelo_path)

if arquivo_csv is not None:
    df = carregar_csv(arquivo_csv)
    st.write("### 🧾 Amostra da base carregada:")
    st.dataframe(df.head())

    with st.spinner("⚙️ Processando e escorando a base..."):
        # Realiza a escoragem usando o pipeline completo do PyCaret
        resultados = predict_model(modelo, data=df, verbose=False)

    st.success("✅ Escoragem concluída com sucesso!")
    st.write("### 🔍 Amostra das previsões:")
    st.dataframe(resultados.head())

    # ==============================================================
    # Botão para download dos resultados
    # ==============================================================
    csv = resultados.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 Baixar resultados (CSV)",
        data=csv,
        file_name="scores_resultados.csv",
        mime="text/csv"
    )

else:
    st.info("Envie um arquivo CSV para iniciar a escoragem.")
