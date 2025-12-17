import pandas as pd

def load_movies(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def load_ratings(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def build_seen_dict(ratings: pd.DataFrame) -> dict:
    # userId -> set(movieId)
    seen = {}
    for row in ratings[["userId", "movieId"]].itertuples(index=False):
        u, m = int(row.userId), int(row.movieId)
        if u not in seen:
            seen[u] = set()
        seen[u].add(m)
    return seen
