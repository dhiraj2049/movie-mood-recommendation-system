"""
app.py

FastAPI backend for the VIBE Movie Recommender.

Run locally:
    uvicorn app:app --reload --port 8000
"""

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model import VibeRecommender, VIBES


# --------------------------------------------------
# Basic setup
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VIBE Movie Recommender",
    version="1.0.0",
    description="Discover movies based on mood, atmosphere, and cinematic vibe."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Paths / Globals
# --------------------------------------------------

MODEL_FILE = Path("vibe_model.pkl")
DATA_FILE = Path("movies.json")
STATIC_DIR = Path("static")

model_instance: Optional[VibeRecommender] = None
movies_index = {}


# --------------------------------------------------
# Response Models
# --------------------------------------------------

class MovieCard(BaseModel):
    id: int
    title: str
    overview: str = ""
    tagline: str = ""
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    vote_average: float = 0.0
    release_date: str = ""
    primary_vibe: str = ""
    similarity_score: float = 0.0
    keyword_names: list[str] = []


class VibeInfo(BaseModel):
    key: str
    label: str
    emoji: str
    color: str
    description: str
    seed_words: str


class RecommendationResponse(BaseModel):
    vibe: str
    vibe_label: str
    vibe_emoji: str
    vibe_color: str
    vibe_description: str
    movies: list[MovieCard]
    total: int


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_model() -> VibeRecommender:
    """
    Loads saved model if available.
    If not found, trains using movies.json.
    """
    global model_instance, movies_index

    if model_instance is not None:
        return model_instance

    if MODEL_FILE.exists():
        logger.info("Loading trained model...")
        model_instance = VibeRecommender.load(str(MODEL_FILE))

    elif DATA_FILE.exists():
        logger.info("No saved model found. Training from movies.json...")

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            movies = json.load(f)

        model_instance = VibeRecommender()
        model_instance.fit(movies)
        model_instance.save(str(MODEL_FILE))

    else:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model files not found.\n"
                "Run:\n"
                "1. python fetch_data.py --api-key YOUR_KEY\n"
                "2. python model.py"
            )
        )

    movies_index = {movie["id"]: movie for movie in model_instance.movies}
    return model_instance


def validate_vibe(vibe: str):
    if vibe not in VIBES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown vibe '{vibe}'. Available: {list(VIBES.keys())}"
        )


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "VIBE backend is running"
    }


@app.get("/vibes", response_model=list[VibeInfo])
def get_vibes():
    """Return all supported vibes."""
    return [
        VibeInfo(key=key, **info)
        for key, info in VIBES.items()
    ]


@app.get("/recommend/{vibe}", response_model=RecommendationResponse)
def recommend_movies(
    vibe: str,
    n: int = Query(12, ge=1, le=50),
    page: int = Query(1, ge=1),
    min_rating: float = Query(5.5, ge=0, le=10),
    seed_movie_id: Optional[int] = None
):
    """
    Get recommendations for a selected vibe.
    Supports pagination.
    """
    validate_vibe(vibe)

    model = load_model()

    fetch_count = n * page

    results = model.recommend(
        vibe=vibe,
        n=fetch_count,
        min_rating=min_rating,
        seed_movie_id=seed_movie_id
    )

    start = (page - 1) * n
    end = page * n
    page_results = results[start:end]

    vibe_meta = VIBES[vibe]

    return RecommendationResponse(
        vibe=vibe,
        vibe_label=vibe_meta["label"],
        vibe_emoji=vibe_meta["emoji"],
        vibe_color=vibe_meta["color"],
        vibe_description=vibe_meta["description"],
        movies=[MovieCard(**movie) for movie in page_results],
        total=len(results)
    )


@app.get("/movie/{movie_id}")
def get_movie(movie_id: int):
    """Get full details for one movie."""
    load_model()

    movie = movies_index.get(movie_id)

    if not movie:
        raise HTTPException(
            status_code=404,
            detail="Movie not found"
        )

    return movie


@app.get("/similar/{movie_id}")
def similar_movies(
    movie_id: int,
    n: int = Query(8, ge=1, le=20)
):
    """Find movies similar to a given title."""
    model = load_model()

    results = model.similar_movies(movie_id, n)

    if not results:
        raise HTTPException(
            status_code=404,
            detail="Movie not found in dataset"
        )

    return {
        "movie_id": movie_id,
        "similar": results
    }


@app.post("/search")
def search_movies(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50)
):
    """Simple title search."""
    load_model()

    query = q.lower().strip()

    matches = [
        movie for movie in movies_index.values()
        if query in movie.get("title", "").lower()
    ]

    matches.sort(
        key=lambda x: x.get("vote_count", 0),
        reverse=True
    )

    return {
        "query": q,
        "results": matches[:limit],
        "total": len(matches)
    }


# --------------------------------------------------
# Frontend
# --------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/app")
    def frontend():
        return FileResponse("static/index.html")