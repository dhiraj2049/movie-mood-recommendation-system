"""
fetch_data.py

Downloads movie data from TMDB and prepares
a dataset for the VIBE recommender.

Usage:
python fetch_data.py --api-key YOUR_KEY
"""

import json
import time
import argparse
import logging
from pathlib import Path

import requests


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Config
# --------------------------------------------------

BASE_URL = "https://api.themoviedb.org/3"
OUTPUT_FILE = "movies.json"


# --------------------------------------------------
# Vibe Keywords
# --------------------------------------------------

VIBE_RULES = {
    "gritty_raw": [
        "crime", "violence", "gang", "street",
        "drugs", "war", "survival", "brutal"
    ],

    "neon_cyberpunk": [
        "future", "robot", "ai", "cyber",
        "technology", "dystopia", "virtual"
    ],

    "noir_dark": [
        "detective", "murder", "mystery",
        "revenge", "dark", "obsession"
    ],

    "offbeat_quirky": [
        "quirky", "weird", "absurd",
        "eccentric", "indie", "odd"
    ],

    "feel_good_warm": [
        "family", "friendship", "hope",
        "love", "joy", "uplifting"
    ],

    "surreal_dreamlike": [
        "dream", "surreal", "mind",
        "abstract", "subconscious",
        "reality", "vision"
    ]
}


# --------------------------------------------------
# TMDB Helpers
# --------------------------------------------------

def tmdb_get(endpoint: str, api_key: str, params=None):
    """
    Generic GET request to TMDB.
    """

    if params is None:
        params = {}

    params["api_key"] = api_key

    url = f"{BASE_URL}/{endpoint}"

    response = requests.get(url, params=params, timeout=15)

    response.raise_for_status()

    return response.json()


def get_popular_movies(api_key: str, pages: int = 5):
    """
    Fetch popular movies page by page.
    """

    all_movies = []

    for page in range(1, pages + 1):
        logger.info(f"Fetching page {page}/{pages}...")

        data = tmdb_get(
            "movie/popular",
            api_key,
            {"page": page}
        )

        all_movies.extend(data.get("results", []))

        time.sleep(0.2)

    return all_movies


def get_keywords(movie_id: int, api_key: str):
    """
    Fetch keywords for a movie.
    """

    try:
        data = tmdb_get(
            f"movie/{movie_id}/keywords",
            api_key
        )

        keywords = data.get("keywords", [])

        return [
            item["name"].lower()
            for item in keywords
            if "name" in item
        ]

    except Exception:
        return []


def get_movie_details(movie_id: int, api_key: str):
    """
    Get full movie details.
    """

    try:
        return tmdb_get(
            f"movie/{movie_id}",
            api_key
        )

    except Exception:
        return {}


# --------------------------------------------------
# Vibe Scoring
# --------------------------------------------------

def score_vibes(text: str):
    """
    Basic keyword scoring system.
    """

    text = text.lower()

    scores = {}

    for vibe, words in VIBE_RULES.items():
        score = sum(
            1 for word in words
            if word in text
        )
        scores[vibe] = score

    return scores


def detect_primary_vibe(movie: dict):
    """
    Decide best vibe for movie.
    """

    content = " ".join([
        movie.get("title", ""),
        movie.get("overview", ""),
        movie.get("tagline", ""),
        " ".join(movie.get("keyword_names", []))
    ])

    scores = score_vibes(content)

    best_vibe = max(
        scores,
        key=scores.get
    )

    return best_vibe, scores


# --------------------------------------------------
# Main Processing
# --------------------------------------------------

def build_dataset(api_key: str, pages: int):
    """
    Download and enrich movie data.
    """

    raw_movies = get_popular_movies(
        api_key,
        pages
    )

    dataset = []

    total = len(raw_movies)

    for i, movie in enumerate(raw_movies, start=1):

        movie_id = movie["id"]

        logger.info(f"[{i}/{total}] {movie.get('title')}")

        details = get_movie_details(
            movie_id,
            api_key
        )

        keywords = get_keywords(
            movie_id,
            api_key
        )

        item = {
            "id": movie_id,
            "title": movie.get("title", ""),
            "overview": movie.get("overview", ""),
            "tagline": details.get("tagline", ""),
            "poster_path": movie.get("poster_path"),
            "backdrop_path": movie.get("backdrop_path"),
            "vote_average": movie.get("vote_average", 0),
            "vote_count": movie.get("vote_count", 0),
            "release_date": movie.get("release_date", ""),
            "genre_ids": movie.get("genre_ids", []),
            "keyword_names": keywords
        }

        primary_vibe, vibe_scores = detect_primary_vibe(item)

        item["primary_vibe"] = primary_vibe
        item["vibe_scores"] = vibe_scores

        dataset.append(item)

        time.sleep(0.15)

    return dataset


def save_dataset(data, path=OUTPUT_FILE):
    """
    Save dataset as JSON.
    """

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    logger.info(f"Saved {len(data)} movies to {path}")


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--api-key",
        required=True,
        help="TMDB API key"
    )

    parser.add_argument(
        "--pages",
        type=int,
        default=5,
        help="Number of pages to fetch"
    )

    args = parser.parse_args()

    logger.info("Starting data fetch...")

    movies = build_dataset(
        api_key=args.api_key,
        pages=args.pages
    )

    save_dataset(movies)

    logger.info("Done.")


if __name__ == "__main__":
    main()