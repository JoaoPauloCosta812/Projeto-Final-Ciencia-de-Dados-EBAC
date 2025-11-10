# Aplicativo de Escoragem de Crédito
### Projeto Final — Curso de Ciência de Dados | EBAC × Semantix

![Streamlit](https://img.shields.io/badge/Streamlit-1.39+-FF4B4B?logo=streamlit&logoColor=white&style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white&style=for-the-badge)
![PyCaret](https://img.shields.io/badge/PyCaret-3.3+-F8C200?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAAAtklEQVQ4T92UsQ3CMAxF34XjKkoTwjAEawAjEawAjABjABjACcZImzVfsgn22bpTtxTt6K1YcV5coVqXPUv5mGfB+R1qCMYMYbcyYmMPmCBhJbU4zNsOK8sBWWdQ1gyXjvC8Sg+JwA/1r1VikPKvMS3vX2DaWwVQZp7dQ0iE+ANChyHhoE8vsscxBwn8BwTAvYt8nHCcqJp7FdpGJYifczpmgAAAABJRU5ErkJggg==&style=for-the-badge)

---

## Sobre o Projeto

Este projeto faz parte do **Módulo Final de Ciência de Dados da EBAC**, em parceria com a **Semantix**.  
O objetivo é desenvolver um **aplicativo interativo de escoragem de crédito** utilizando *Machine Learning*, que permita:
- Carregar uma nova base de clientes;
- Processar automaticamente as variáveis preditoras;
- Aplicar o modelo de crédito treinado (`model_final.pkl`);
- Exibir **scores de inadimplência**, métricas e gráficos interativos.

O aplicativo foi desenvolvido em **Streamlit** e pode ser executado tanto localmente quanto no **Streamlit Cloud**.

---

## Demonstração

🔗 **Aplicação Online:**  
[projeto-final-ciencia-de-dados-ebac.streamlit.app](https://projeto-final-ciencia-de-dados-ebac.streamlit.app)



https://github.com/user-attachments/assets/df6241cc-4f94-41d7-8da8-7edfdcde93b4



---

## Funcionalidades Principais

- 📂 **Upload de Base CSV** para novas escoragens;
- ⚙️ **Processamento automático** de tipos, colunas e variáveis categóricas;
- 🧮 **Aplicação do modelo PyCaret** salvo em `model_final.pkl`;
- 📊 **Gráficos interativos** com Plotly:
  - Distribuição dos scores (histograma);
  - Proporção de aprovações e reprovações (gráfico de pizza);
- 📈 **Métricas resumo** (score médio, % aprovados, % reprovados);
- 💾 **Download dos resultados** com scores e classificações em CSV.
