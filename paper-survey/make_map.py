import numpy as np
import pandas as pd
import plotly.express as px
import umap

df = pd.read_csv("papers.csv")
vectors = np.load("vectors.npy")

# 384次元 → 2次元に縮める
reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
coords = reducer.fit_transform(vectors)
df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

# マウスで触れる散布図を作る
fig = px.scatter(
    df,
    x="x",
    y="y",
    hover_name="title",
    hover_data={"published": True, "x": False, "y": False},
    title="論文マップ(近い点ほど内容が似ている)",
)
fig.update_traces(marker=dict(size=7, opacity=0.7))
fig.write_html("map.html")
fig.show()