from app.schemas.recommend import Movie

def predict_movie(movie: Movie) -> dict:
    score = 0

    # règle simple
    if movie.runtime >= 120:
        score += 1

    if len(movie.genres) >= 3:
        score += 1

    label = "RECOMMENDED" if score >= 1 else "NOT_RECOMMENDED"

    return {"label": label, "score": score}
