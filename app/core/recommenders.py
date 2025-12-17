import numpy as np
def get_movie_id_col(movies):
    # Priorité MovieLens -> TMDB
    for col in ["movieId", "id", "tmdbId"]:
        if col in movies.columns:
            return col
    raise KeyError(f"Aucune colonne ID trouvée dans movies. Colonnes: {list(movies.columns)}")

def recommend_cf(user_id, svd_model, ratings, movies, topk=10):
    user_id = int(user_id)
    if user_id not in ratings["userId"].unique():
        return []

    seen_movies = set(ratings[ratings["userId"] == user_id]["movieId"])
    all_movies = movies["movieId"].dropna().astype(int).unique()
    candidates = [m for m in all_movies if m not in seen_movies]
    if not candidates:
        return []

    preds = []
    for m in candidates:
        est = svd_model.predict(user_id, int(m)).est
        preds.append((int(m), float(est)))

    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:topk]

    # join title
    id_to_title = dict(zip(movies["movieId"].astype(int), movies["title"].astype(str)))
    return [{"movieId": mid, "title": id_to_title.get(mid, ""), "score_pred": score} for mid, score in top]


def recommend_logistic_for_user(user_id, clf, enc, ratings, movies, topk=10, min_prob=0.5):
    user_id = int(user_id)
    if user_id not in ratings["userId"].unique():
        return []

    seen_movies = set(ratings[ratings["userId"] == user_id]["movieId"])
    all_movies = movies["movieId"].dropna().astype(int).unique()
    candidates = [m for m in all_movies if m not in seen_movies]
    if not candidates:
        return []

    # construire X candidates : (userId, movieId)
    Xc = enc.transform(np.column_stack([np.full(len(candidates), user_id), np.array(candidates)]))
    probs = clf.predict_proba(Xc)[:, 1]

    recs = [(int(m), float(p)) for m, p in zip(candidates, probs) if p >= min_prob]
    recs.sort(key=lambda x: x[1], reverse=True)
    top = recs[:topk]

    id_to_title = dict(zip(movies["movieId"].astype(int), movies["title"].astype(str)))
    return [{"movieId": mid, "title": id_to_title.get(mid, ""), "score_log": score} for mid, score in top]


def hybrid_recommend_for_user(user_id, clf, enc, svd_model, ratings, movies, seen_dict, topk=10, alpha=0.5):
    user_id = int(user_id)
    if user_id not in ratings["userId"].unique():
        return []

    seen = seen_dict.get(user_id, set())
    all_movies = movies["movieId"].dropna().astype(int).unique()
    candidates = [m for m in all_movies if m not in seen]
    if not candidates:
        return []

    # logistic prob
    Xc = enc.transform(np.column_stack([np.full(len(candidates), user_id), np.array(candidates)]))
    probs = clf.predict_proba(Xc)[:, 1]  # 0..1

    # svd note (0.5..5) -> normaliser en 0..1
    svd_scores = np.array([svd_model.predict(user_id, int(m)).est for m in candidates])
    svd_norm = (svd_scores - 0.5) / (5.0 - 0.5)

    hybrid = alpha * probs + (1 - alpha) * svd_norm

    # top
    idx = np.argsort(-hybrid)[:topk]
    id_to_title = dict(zip(movies["movieId"].astype(int), movies["title"].astype(str)))

    out = []
    for i in idx:
        mid = int(candidates[i])
        out.append({
            "movieId": mid,
            "title": id_to_title.get(mid, ""),
            "hybrid_score": float(hybrid[i]),
            "score_log": float(probs[i]),
            "score_svd": float(svd_scores[i]),
        })
    return out
