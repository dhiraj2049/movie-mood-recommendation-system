"""
model.py

Core recommendation engine for the VIBE movie recommender.
Uses TF-IDF text vectors + cosine similarity.
"""

import json
import pickle
import logging
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Vibe Definitions
# --------------------------------------------------

VIBES = {
    "gritty_raw": {
        "label": "Gritty & Raw",
        "emoji": "🔥",
        "color": "#c0392b",
        "description": "Street-level stories with realism and intensity.",
        "seed_words": "crime brutal street urban violence gang poverty struggle"
    },

    "neon_cyberpunk": {
        "label": "Neon & Cyberpunk",
        "emoji": "⚡",
        "color": "#00d2ff",
        "description": "Digital futures, neon cities, rebellion.",
        "seed_words": "future dystopia robot ai hacker cyberpunk neon technology"
    },

    "noir_dark": {
        "label": "Noir & Dark",
        "emoji": "🌑",
        "color": "#8e44ad",
        "description": "Mystery, shadows, obsession and danger.",
        "seed_words": "detective mystery revenge corruption murder suspense dark noir"
    },

    "offbeat_quirky": {
        "label": "Offbeat & Quirky",
        "emoji": "🎭",
        "color": "#f39c12",
        "description": "Unusual, eccentric and charming stories.",
        "seed_words": "quirky absurd indie weird whimsical odd unconventional"
    },

    "feel_good_warm": {
        "label": "Feel-Good & Warm",
        "emoji": "🌻",
        "color": "#27ae60",
        "description": "Heartwarming films that uplift.",
        "seed_words": "family friendship joy uplifting redemption wholesome love hope"
    },

    "surreal_dreamlike": {
        "label": "Surreal & Dreamlike",
        "emoji": "🌀",
        "color": "#e91e8c",
        "description": "Reality bends into poetic cinema.",
        "seed_words": "dream surreal subconscious metaphysical abstract mind bending ethereal"
    }
}


# --------------------------------------------------
# Genre Mapping
# --------------------------------------------------

GENRES = {
    28: "action",
    12: "adventure",
    16: "animation",
    35: "comedy",
    80: "crime",
    18: "drama",
    10751: "family",
    14: "fantasy",
    27: "horror",
    9648: "mystery",
    878: "science fiction",
    53: "thriller",
    10752: "war"
}


# --------------------------------------------------
# Text Builder
# --------------------------------------------------

def build_movie_text(movie: dict) -> str:
    """
    Combine overview, tagline, genres, keywords and vibe words
    into one searchable text profile.
    """

    parts = []

    overview = movie.get("overview", "")
    tagline = movie.get("tagline", "")

    if overview:
        parts.append(overview)

    if tagline:
        parts.extend([tagline, tagline])

    genre_ids = movie.get("genre_ids", [])

    for gid in genre_ids:
        if gid in GENRES:
            parts.extend([GENRES[gid]] * 2)

    for keyword in movie.get("keyword_names", []):
        parts.append(keyword)

    vibe = movie.get("primary_vibe")

    if vibe in VIBES:
        parts.append(VIBES[vibe]["seed_words"])

    return " ".join(parts).lower()


# --------------------------------------------------
# Main Recommender
# --------------------------------------------------

class VibeRecommender:

    def __init__(self):
        self.movies = []
        self.id_to_index = {}

        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )

        self.matrix = None
        self.vibe_vectors = {}
        self.ready = False

    # ----------------------------------------------

    def fit(self, movies: list):
        """
        Train model on movie dataset.
        """
        logger.info("Preparing training data...")

        self.movies = movies
        texts = [build_movie_text(movie) for movie in movies]

        self.matrix = self.vectorizer.fit_transform(texts)
        self.id_to_index = {
            movie["id"]: i for i, movie in enumerate(movies)
        }

        for vibe_key, vibe_data in VIBES.items():
            self.vibe_vectors[vibe_key] = self.vectorizer.transform(
                [vibe_data["seed_words"]]
            )

        self.ready = True
        logger.info("Model ready.")

    # ----------------------------------------------

    def recommend(
        self,
        vibe: str,
        n: int = 12,
        min_rating: float = 5.5,
        seed_movie_id: int = None
    ) -> list:

        if not self.ready:
            raise RuntimeError("Model not trained.")

        if vibe not in self.vibe_vectors:
            raise ValueError("Unknown vibe.")

        query = self.vibe_vectors[vibe]

        # Blend with seed movie
        if seed_movie_id in self.id_to_index:
            idx = self.id_to_index[seed_movie_id]
            movie_vector = self.matrix[idx]
            query = (0.6 * movie_vector) + (0.4 * query)

        similarities = cosine_similarity(
            query,
            self.matrix
        ).flatten()

        candidates = []

        for i, score in enumerate(similarities):
            movie = self.movies[i]

            if movie["id"] == seed_movie_id:
                continue

            if movie.get("vote_average", 0) < min_rating:
                continue

            if not movie.get("poster_path"):
                continue

            candidates.append({
                "movie": movie,
                "similarity": score
            })

        if not candidates:
            return []

        ratings = np.array([
            c["movie"].get("vote_average", 0)
            for c in candidates
        ])

        scores = np.array([
            c["similarity"]
            for c in candidates
        ])

        scaler = MinMaxScaler()

        rating_scaled = scaler.fit_transform(
            ratings.reshape(-1, 1)
        ).flatten()

        score_scaled = scaler.fit_transform(
            scores.reshape(-1, 1)
        ).flatten()

        final_scores = (
            0.65 * score_scaled +
            0.35 * rating_scaled
        )

        for i, candidate in enumerate(candidates):
            candidate["final_score"] = final_scores[i]

        ranked = sorted(
            candidates,
            key=lambda x: x["final_score"],
            reverse=True
        )[:n]

        results = []

        for item in ranked:
            movie = item["movie"]

            results.append({
                "id": movie["id"],
                "title": movie.get("title", ""),
                "overview": movie.get("overview", ""),
                "tagline": movie.get("tagline", ""),
                "poster_path": movie.get("poster_path"),
                "backdrop_path": movie.get("backdrop_path"),
                "vote_average": round(movie.get("vote_average", 0), 1),
                "vote_count": movie.get("vote_count", 0),
                "release_date": movie.get("release_date", ""),
                "primary_vibe": movie.get("primary_vibe", vibe),
                "keyword_names": movie.get("keyword_names", [])[:8],
                "similarity_score": round(
                    float(item["final_score"]), 4
                )
            })

        return results

    # ----------------------------------------------

    def similar_movies(self, movie_id: int, n: int = 8):
        """
        Return content-similar movies.
        """

        if movie_id not in self.id_to_index:
            return []

        idx = self.id_to_index[movie_id]

        vector = self.matrix[idx]

        scores = cosine_similarity(
            vector,
            self.matrix
        ).flatten()

        best = np.argsort(scores)[::-1][1:n + 1]

        return [
            self.movies[i]
            for i in best
        ]

    # ----------------------------------------------

    def save(self, path="vibe_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)

        logger.info("Model saved.")

    @classmethod
    def load(cls, path="vibe_model.pkl"):
        with open(path, "rb") as f:
            model = pickle.load(f)

        logger.info("Model loaded.")
        return model


# --------------------------------------------------
# CLI Training
# --------------------------------------------------

if __name__ == "__main__":

    file_path = Path("movies.json")

    if not file_path.exists():
        print("movies.json not found.")
        exit()

    with open(file_path, "r", encoding="utf-8") as f:
        movies = json.load(f)

    engine = VibeRecommender()
    engine.fit(movies)
    engine.save()

    print("Training complete.")