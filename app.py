import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Score de Crédito",
    page_icon="💳",
    layout="wide",
)
st.title("💳 Aplicativo de Escoragem de Crédito")
st.caption("Utilize este app para escorar novas bases com o modelo treinado (`model_final.pkl`).")


# -----------------------------------------------------------------------------
# 1. CARREGAR MODELO TREINADO (o .pkl que você subiu pro GitHub)
# -----------------------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    # o nome tem que ser exatamente o que está no repositório
    return load_model("model_final")

modelo = carregar_modelo()


# -----------------------------------------------------------------------------
# 2. FUNÇÃO DE LIMPEZA / NORMALIZAÇÃO DA BASE QUE O USUÁRIO FAZ UPLOAD
# -----------------------------------------------------------------------------
def preparar_base_para_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 2.1 – remover colunas de índice que vieram do to_csv
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed") or c.lower() == "index"]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 2.2 – no score não mandamos a resposta
    if "mau" in df.columns:
        df = df.drop(columns=["mau"])

    # 2.3 – converter data_ref pra datetime (no treino ela existia assim)
    if "data_ref" in df.columns:
        df["data_ref"] = pd.to_datetime(df["data_ref"], errors="coerce")

    # 2.4 – colunas numéricas que às vezes vêm com ponto e vírgula
    colunas_numericas_suspeitas = [
        "renda",
        "tempo_emprego",
        "idade",
        "qt_pessoas_residencia",
        "qtd_filhos",
    ]
    for col in colunas_numericas_suspeitas:
        if col in df.columns:
            # vira string, tira separador de milhar, troca vírgula por ponto e converte
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2.5 – qualquer coisa que ficar NaN a gente deixa como está; o pipeline do PyCaret lida
    return df


# -----------------------------------------------------------------------------
# 3. SIDEBAR – UPLOAD
# -----------------------------------------------------------------------------
st.sidebar.header("📂 Upload de Base")
arquivo = st.sidebar.file_uploader("Envie um arquivo CSV", type=["csv"])

if arquivo is None:
    st.info("Envie um arquivo CSV para iniciar a escoragem.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. LER E MOSTRAR A BASE
# -----------------------------------------------------------------------------
df_raw = pd.read_csv(arquivo)
st.write("### 🧾 Amostra da base carregada:")
st.dataframe(df_raw.head())


# -----------------------------------------------------------------------------
# 5. PREPARAR BASE E RODAR O MODELO
# -----------------------------------------------------------------------------
df_ready = preparar_base_para_score(df_raw)

try:
    # o predict_model do PyCaret já devolve o dataframe + colunas de previsão
    resultados = predict_model(modelo, data=df_ready, verbose=False)

    st.success("✅ Escoragem concluída!")
    st.write("### 🔍 Amostra das previsões:")
    st.dataframe(resultados.head())

    # botão para download
    csv_out = resultados.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 Baixar resultados (CSV)",
        data=csv_out,
        file_name="scores_resultados.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error("❌ Não consegui escorar essa base com o modelo atual.")
    st.write(
        "Isso geralmente acontece quando **as colunas do CSV não estão no mesmo formato** "
        "que o modelo foi treinado (ex: data como texto, coluna extra, ou target junto)."
    )
    st.write("**Mensagem técnica (pode mostrar pro professor):**")
    st.code(str(e))
    st.write("**dtypes recebidos:**")
    st.dataframe(df_ready.dtypes)
