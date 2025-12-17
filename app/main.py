from fastapi import FastAPI
from app.routers.recommend import router as recommend_router
import pandas as pd
from app.core.state import MODELS
from app.core.data_loader import load_movies, load_ratings, build_seen_dict
from app.core.train_models import train_svd, train_logistic
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="MovieMood API")

@app.on_event("startup")
def startup():
    # 1) charger data
    movies = load_movies("data/movies_metadata.csv")
    ratings = load_ratings("data/ratings_small.csv")
    movies["movieId"] = pd.to_numeric(movies["id"], errors="coerce")
    movies = movies.dropna(subset=["movieId"])
    movies["movieId"] = movies["movieId"].astype(int)
    # IMPORTANT: s'assurer d'avoir movieId/userId en int si besoin
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)

    # 2) seen_dict
    seen_dict = build_seen_dict(ratings)

    # 3) entraîner modèles en mémoire
    svd = train_svd(ratings)
    clf, enc = train_logistic(ratings)

    # 4) stocker
    MODELS["movies"] = movies
    MODELS["ratings"] = ratings
    MODELS["seen_dict"] = seen_dict
    MODELS["svd"] = svd
    MODELS["clf"] = clf
    MODELS["enc"] = enc
    MODELS["ready"] = True





app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend_router)
