# CineGraph

CineGraph is a lightweight knowledge-graph + transformer hybrid movie recommender and demo UI built on a small TMDB dataset. It scores movies by shared factual relations (genres, keywords, production companies, named entities, etc.) and optionally reranks using sentence-transformer embeddings.

## What this is
A hybrid recommendation demo that combines a weighted knowledge graph of movie facts with an optional transformer reranker to surface interpretable movie recommendations and an SVG-based graph reasoning view.

## Stack
- Languages: Python (FastAPI) + HTML/vanilla JS frontend
- Framework / runtime: FastAPI backend, static HTML UI (index.html)
- Notable libraries: pandas, numpy, spaCy, sentence-transformers (optional), scikit-learn (cosine similarity)

## How it's organized
```
app.py                   # FastAPI app: builds graph, embeddings (optional), exposes /search & /recommend
index.html               # Front-end demo UI (static) that calls the backend and renders the SVG graph
tmdb_5000.csv            # TMDB dataset (subset used by the demo)
movie_vectors*.npy       # Precomputed embedding / vector files (cache)
movie_graph_transformer_embeddings.npy  # Transformer embeddings cache
__pycache__/             # Python cache files (ignored normally)
```

How it fits together: On startup app.py loads `tmdb_5000.csv`, extracts factual concepts per movie (genres, keywords, companies, entities from overviews, etc.), builds an inverted relation index and (optionally) sentence-transformer embeddings. The backend exposes a simple search endpoint and a /recommend endpoint which returns recommended movies and a knowledge-graph payload used by the frontend to render an interpretable graph.

## Requirements
- Python 3.8+
- Recommended packages: fastapi, uvicorn, pandas, numpy, spacy, sentence-transformers, scikit-learn

You can install the main dependencies with:

```bash
python -m pip install --upgrade pip
pip install fastapi uvicorn pandas numpy spacy scikit-learn sentence-transformers
python -m spacy download en_core_web_sm
```

Note: sentence-transformers and scikit-learn are optional — if `sentence-transformers` is not installed the app will run in graph-only mode.

## How to run (development)
1. Clone the repo and cd into it.
2. Ensure the dataset `tmdb_5000.csv` is present (it is included in this repo).
3. Run the FastAPI app:

```bash
# option A: run directly
python app.py

# option B: use uvicorn
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

4. Open `index.html` in a browser (double-click or serve it with a simple static server) and use the frontend. The frontend expects the backend at http://127.0.0.1:8000 by default.

Important environment/keys
- The frontend uses OMDB to fetch posters. Edit `index.html` and replace the `OMDB_KEY` variable near the top of the file with your OMDB API key to show movie posters. If you don't have a key, the UI will still work but posters will fall back to placeholders.

## Endpoints
- `GET /search?query=...` — returns matched movie titles (autocomplete)
- `GET /recommend?title=...` — returns recommendations and a `graph` payload used by the UI

Example request:
```bash
curl "http://127.0.0.1:8000/recommend?title=Avatar"
```

## Notes & Tips
- The first startup may take longer if sentence-transformers is enabled because embeddings are computed and saved to `movie_graph_transformer_embeddings.npy`.
- The app includes logic to download the spaCy `en_core_web_sm` model automatically if it's not present, but it's recommended to install it manually in virtual environments.
- The repository includes several large binary files (`*.npy`, the CSV). If you plan to trim the repo for distribution, consider removing embedding caches and the dataset and downloading them outside the repo instead.

## How you can help / TODOs
- Add a `requirements.txt` or `pyproject.toml` for reproducible installs.
- Serve `index.html` from the FastAPI app (static files) so the demo works without opening the file directly.
- Add tests for the recommender functions (recommend_from_graph, relation_specificity).
- Add CLI flags to control LIMIT and embedding behavior.

## License
MIT License — feel free to reuse and adapt.

---

If you want, I can also:
- add a requirements.txt and a minimal GitHub Actions workflow to run linting/tests,
- serve the frontend from the FastAPI app, or
- update index.html to read the OMDB key from an environment variable or query param.
