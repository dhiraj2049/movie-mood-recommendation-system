# 🎬 VIBE — Movie Recommender

A vibe-based movie recommendation system using TF-IDF + cosine similarity,
powered by real TMDB data. Discover films by their emotional feel — not just genre.

## Supported Vibes

| Vibe | Description |
|------|-------------|
| 🔥 Gritty & Raw | Unflinching, street-level stories with brutal honesty |
| ⚡ Neon & Cyberpunk | Electric futures, neon-soaked cities, digital rebellion |
| 🌑 Noir & Dark | Moody shadows, moral ambiguity, dangerous obsessions |
| 🎭 Offbeat & Quirky | Strange, wonderful worlds that play by their own rules |
| 🌻 Feel-Good & Warm | Movies that leave you smiling and hopeful |
| 🌀 Surreal & Dreamlike | Reality bends — pure cinematic poetry |

---

## Setup (3 steps)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get your TMDB API key
1. Sign up free at https://www.themoviedb.org/signup
2. Go to **Settings → API → Create → Developer**
3. Copy your **API Key (v3 auth)**

### 3. Fetch movie data
```bash
python fetch_data.py --api-key YOUR_TMDB_API_KEY --pages 5
```
This fetches ~1,500–2,000 movies and saves them to `movies.json`.
It takes about 3–5 minutes.

### 4. Train the model
```bash
python model.py --data movies.json --output vibe_model.pkl
```

### 5. Start the server
```bash
uvicorn app:app --reload --port 8000
```

### 6. Open the UI
Open `static/index.html` in your browser, OR visit:
http://localhost:8000/app

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/vibes` | List all supported vibes |
| GET | `/recommend/{vibe}` | Get recommendations for a vibe |
| GET | `/similar/{movie_id}` | Find similar movies |
| GET | `/movie/{movie_id}` | Movie details |
| POST | `/search?q=title` | Search by title |

### Example
```bash
# Get 12 noir recommendations
curl http://localhost:8000/recommend/noir_dark?n=12

# Get movies similar to The Dark Knight (movie_id: 155)
curl http://localhost:8000/similar/155
```

---

## Architecture

```
fetch_data.py         → Pulls movies from TMDB API, enriches with keywords
       ↓
movies.json           → ~2,000 enriched movie records

model.py              → VibeRecommender
  ├── Feature Engineering
  │     overview + tagline + genres + keywords → vibe_text
  ├── TF-IDF Vectorizer (15k features, bigrams)
  ├── Cosine Similarity (query vector vs all movies)
  ├── Vibe Anchor Vectors (seed words per vibe)
  └── Re-ranking (0.65 × similarity + 0.35 × vote score)
       ↓
vibe_model.pkl        → Serialized trained model

app.py                → FastAPI REST API
static/index.html     → Cinematic dark frontend UI
```

## Customising Vibes

Edit the `VIBES` dict in `model.py` to add/change vibes:

```python
VIBES["cosmic_horror"] = {
    "label": "Cosmic Horror",
    "emoji": "👁️",
    "color": "#1a1a2e",
    "seed_words": "lovecraftian eldritch cosmic horror unknown entity ancient",
    "description": "Unknowable terrors from beyond human comprehension",
}
```

Then retrain:
```bash
python model.py
```
