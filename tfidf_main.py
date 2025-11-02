import os, pickle, numpy as np, pandas as pd, gc, re
from tqdm import tqdm
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from collections import defaultdict

# -----------------------------
# 0) 경로
# -----------------------------
YELP_CLEAN_PATH = "data/train/clean_yelp.pkl"
OT_CLEAN_PATH   = "data/test/clean_opentable.pkl"
ARTIF_DIR = "artifacts_tfidf_memsafe"
os.makedirs(ARTIF_DIR, exist_ok=True)

# -----------------------------
# 1) 데이터 로드
# -----------------------------
df_yelp = pd.read_pickle(YELP_CLEAN_PATH)
df_ot   = pd.read_pickle(OT_CLEAN_PATH)
print(f"Yelp:{df_yelp.shape}, OpenTable:{df_ot.shape}")

# -----------------------------
# 2) TF-IDF
# -----------------------------
print("TF-IDF 전처리 중...")
agg = df_yelp.groupby("business_id").agg({
    "name":"first","categories":"first",
    "review_text":lambda x:" ".join(x.fillna(""))
}).reset_index()
texts = (agg["name"].fillna("") + " " + agg["categories"].fillna("") + " " + agg["review_text"].fillna("")).values
print(f"아이템 수:{len(agg)}")

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,1), min_df=5, stop_words='english', dtype=np.float32)
X_items_yelp = tfidf.fit_transform(texts)
X_items_yelp = normalize(X_items_yelp, norm="l2", copy=False)
print("TF-IDF 행렬:", X_items_yelp.shape)

biz_ids = agg["business_id"].astype(str).values
biz2idx = {b:i for i,b in enumerate(biz_ids)}
idx2biz = {i:b for b,i in biz2idx.items()}

del agg, texts; gc.collect()

# -----------------------------
# 3) 유저 프로필 구축 (메모리 효율)
# -----------------------------
df_yelp = df_yelp.sort_values("user_id")
test_idx = df_yelp.groupby("user_id").tail(1).index
df_train = df_yelp.drop(index=test_idx)
df_test  = df_yelp.loc[test_idx]

POS_TH = 0.6
user_profiles = {}
pos = df_train[df_train["rating_norm"] >= POS_TH]

print("사용자 프로필 계산 중...")
for u, g in tqdm(pos.groupby("user_id"), total=pos["user_id"].nunique()):
    idxs = [biz2idx[b] for b in g["business_id"] if b in biz2idx]
    n = len(idxs)
    if n == 0: continue
    if n > 1000: idxs = np.random.choice(idxs, 1000, replace=False)
    # CSR 행렬에서 직접 합계 계산
    sub = X_items_yelp[idxs]
    mean_vec = sub.mean(axis=0)                          # (1, n_features) np.matrix
    mean_vec = np.asarray(mean_vec).ravel().astype(np.float32)  # -> (n_features,) ndarray
    prof = normalize(mean_vec.reshape(1, -1), norm="l2")        # (1, n_features) ndarray
    user_profiles[u] = prof
    del sub, mean_vec
    if len(user_profiles) % 500 == 0: gc.collect()
    print("Profiles:", len(user_profiles))

# -----------------------------
# 4) 평가 (Top-K 제한 + 샘플 유저)
# -----------------------------
def recall_at_k(targets, ranked, k):
    return 1.0 if any(t in ranked[:k] for t in targets) else 0.0
def ndcg_at_k(targets, ranked, k):
    rels=[1 if x in targets else 0 for x in ranked]
    dcg=np.sum((2**np.array(rels[:k])-1)/np.log2(np.arange(2,len(rels[:k])+2)))
    idcg=np.sum((2**np.sort(rels)[::-1]-1)/np.log2(np.arange(2,len(rels[:k])+2)))
    return dcg/idcg if idcg>0 else 0.0
def apk(targets, ranked, k):
    hits=s=0
    for i,x in enumerate(ranked[:k],1):
        if x in targets: hits+=1; s+=hits/i
    return s/min(len(targets),k) if targets else 0.0

seen=defaultdict(set)
for r in df_train.itertuples(index=False):
    seen[r.user_id].add(str(r.business_id))

TOPK=10
recalls, ndcgs, maps = [],[],[]
sample_users=list(df_test["user_id"].unique())[:1000]  # 상위 1000명만 평가
print("Yelp 평가 중...")
for u in tqdm(sample_users):
    if u not in user_profiles: continue
    tg=df_test[df_test["user_id"]==u]["business_id"].astype(str).tolist()
    up_vec = user_profiles[u].ravel()
    # 희소 행렬 dot-product (dense 변환 없음)
    scores = X_items_yelp.dot(up_vec) 
    for b in seen[u]:
        if b in biz2idx: scores[biz2idx[b]] = -1e9
    top_idx=np.argpartition(-scores,TOPK)[:TOPK]
    ranked=[idx2biz[i] for i in top_idx[np.argsort(-scores[top_idx])]]
    recalls.append(recall_at_k(tg, ranked, TOPK))
    ndcgs.append(ndcg_at_k(tg, ranked, TOPK))
    maps.append(apk(tg, ranked, TOPK))
    if len(recalls)%100==0: gc.collect()

print(f"[Yelp/TF-IDF] Users={len(recalls)} Recall@{TOPK}={np.mean(recalls):.4f} nDCG@{TOPK}={np.mean(ndcgs):.4f} MAP@{TOPK}={np.mean(maps):.4f}")

# -----------------------------
# 5) 전이 테스트 (간략)
# -----------------------------
def build_item_text_ot(df):
    return (df["name"].fillna("")+" "+df["categories"].fillna("")+" "+df["review_text"].fillna("")).values
text_ot=build_item_text_ot(df_ot)
X_items_ot=tfidf.transform(text_ot)
X_items_ot=normalize(X_items_ot, norm="l2", copy=False)
print("OpenTable 행렬:", X_items_ot.shape)

# -----------------------------
# 6) 저장
# -----------------------------
with open(os.path.join(ARTIF_DIR,"vectorizer_tfidf.pkl"),"wb") as f:
    pickle.dump(tfidf,f)
sparse.save_npz(os.path.join(ARTIF_DIR,"X_items_yelp_tfidf.npz"),X_items_yelp)
with open(os.path.join(ARTIF_DIR,"biz_ids.pkl"),"wb") as f:
    pickle.dump(biz_ids,f)
with open(os.path.join(ARTIF_DIR,"user_profiles_tfidf.pkl"),"wb") as f:
    pickle.dump(user_profiles,f)
print("✅ 결과 저장 완료:", ARTIF_DIR)
