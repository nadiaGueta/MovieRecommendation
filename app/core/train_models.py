from surprise import Dataset, Reader, SVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_svd(ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()

    svd = SVD(n_factors=50, n_epochs=20, random_state=42)
    svd.fit(trainset)
    return svd

def train_logistic(ratings):
    # Exemple : label = 1 si rating >= 4, sinon 0 (comme souvent dans notebooks)
    ratings = ratings.copy()
    ratings["label"] = (ratings["rating"] >= 4.0).astype(int)

    enc = OneHotEncoder(handle_unknown="ignore", dtype=float)
    X = enc.fit_transform(ratings[["userId", "movieId"]])
    y = ratings["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    clf = LogisticRegression(solver="liblinear", max_iter=200, class_weight="balanced")
    clf.fit(X_train, y_train)

    return clf, enc
