import arxiv
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ===== ここだけ変えればOK =====
QUERY = "cat:cs.LG AND abs:recommendation"
MAX_RESULTS = 100
# ==============================

# 1. arXivから論文を取得
print("論文を取得中...")
client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
search = arxiv.Search(
    query=QUERY,
    max_results=MAX_RESULTS,
    sort_by=arxiv.SortCriterion.SubmittedDate,
)

papers = []
for r in client.results(search):
    papers.append({
        "title": r.title.replace("\n", " "),
        "abstract": r.summary.replace("\n", " "),
        "published": r.published.strftime("%Y-%m-%d"),
        "url": r.entry_id,
    })

df = pd.DataFrame(papers)
df.to_csv("papers.csv", index=False)
print(f"{len(df)}件取得して papers.csv に保存しました")

# 2. 要約をベクトルに変換
print("ベクトル化中...")
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(
    df["abstract"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True,
)
np.save("vectors.npy", vectors)
print("vectors.npy に保存しました")