import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="論文サーベイ支援", layout="wide")
st.title("論文サーベイ支援ツール")


@st.cache_data
def load_data():
    return pd.read_csv("papers.csv"), np.load("vectors.npy")


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


df, vectors = load_data()
model = load_model()

query = st.text_input(
    "探したい内容を英語で入力してください",
    "graph neural network for recommendation",
)
top_k = st.slider("表示件数", 3, 20, 5)

if query:
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = vectors @ q
    top = scores.argsort()[::-1][:top_k]

    for i in top:
        row = df.iloc[i]
        st.subheader(row["title"])
        st.caption(f"類似度 {scores[i]:.3f}  |  {row['published']}")
        st.write(row["abstract"][:300] + "...")
        st.link_button("arXivで開く", row["url"])
        st.divider()