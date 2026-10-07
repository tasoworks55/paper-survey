import arxiv
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. arXivから論文を100件取得
print("論文を取得中...")
search = arxiv.Search(
    query="cat:cs.LG AND all:graph neural network",
    max_results=100,
    sort_by=arxiv.SortCriterion.SubmittedDate,
)
papers = [
    {"title": r.title, "abstract": r.summary}
    for r in arxiv.Client().results(search)
]
df = pd.DataFrame(papers)
df.to_csv("papers.csv", index=False)
print(f"{len(df)}件取得しました")

# 2. 要約をベクトルに変換(初回はモデルのダウンロードで数分かかります)
print("ベクトル化中...")
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(df["abstract"].tolist())

# 3. 0番目の論文に似ている論文を5件表示
sim = cosine_similarity([vectors[0]], vectors)[0]
top = sim.argsort()[::-1][1:6]

print("\n基準の論文:", df["title"][0])
print("\n似ている論文:")
for i in top:
    print(f"  {sim[i]:.3f}  {df['title'][i]}")