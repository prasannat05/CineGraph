import difflib
import json
import math
import os

import numpy as np
import pandas as pd
import spacy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    SentenceTransformer = None
    cosine_similarity = None

# --- INITIALIZATION ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Configuration
DATA_PATH = "tmdb_5000.csv"
LIMIT = 3000
TOP_K = 6
EMBEDDING_CACHE_PATH = "movie_graph_transformer_embeddings.npy"
RELATION_WEIGHTS = {
    "genres": 4.0,
    "keywords": 3.6,
    "entities": 2.8,
    "companies": 2.2,
    "countries": 1.8,
    "languages": 1.4,
    "original_language": 1.2,
    "decade": 1.1,
    "runtime_band": 1.0,
    "status": 0.7,
}
HYBRID_GRAPH_WEIGHT = 0.72
HYBRID_TRANSFORMER_WEIGHT = 0.28
DISPLAY_LABELS = {
    "genres": "Genre",
    "keywords": "Keyword",
    "entities": "Entity",
    "companies": "Company",
    "countries": "Country",
    "languages": "Language",
    "original_language": "Original Language",
    "decade": "Decade",
    "runtime_band": "Runtime",
    "status": "Status",
}

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

transformer_model = None
transformer_embeddings = None


# --- HELPERS ---
def safe_parse_json(data):
    try:
        if pd.isna(data):
            return ""
        items = json.loads(data)
        return " ".join([item["name"] for item in items if item.get("name")])
    except Exception:
        return ""


def parse_json_names(data):
    try:
        if pd.isna(data):
            return []
        items = json.loads(data)
        return [item["name"].strip() for item in items if item.get("name")]
    except Exception:
        return []


def unique_terms(items):
    seen = set()
    ordered = []
    for item in items:
        if not item:
            continue
        key = str(item).strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(str(item).strip())
    return ordered


def extract_ner_nodes(text):
    if not text or len(str(text)) < 10:
        return []
    doc = nlp(str(text))
    entities = [ent.text.strip() for ent in doc.ents if ent.label_ in ["GPE", "PERSON", "ORG"]]
    return unique_terms(entities)


def parse_year(date_value):
    if not date_value or pd.isna(date_value):
        return None
    text = str(date_value).strip()
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


def runtime_band(runtime):
    try:
        minutes = float(runtime)
    except Exception:
        return None
    if minutes < 90:
        return "Short Runtime"
    if minutes <= 130:
        return "Feature Runtime"
    return "Epic Runtime"


def relation_id(relation_type, value):
    return f"{relation_type}::{value}"


def build_movie_concepts(row):
    genres = parse_json_names(row.get("genres", ""))
    keywords = parse_json_names(row.get("keywords", ""))
    companies = parse_json_names(row.get("production_companies", ""))
    countries = parse_json_names(row.get("production_countries", ""))
    languages = parse_json_names(row.get("spoken_languages", ""))
    entities = extract_ner_nodes(row.get("overview", ""))

    release_year = parse_year(row.get("release_date"))
    decade = f"{(release_year // 10) * 10}s" if release_year else None
    original_language = row.get("original_language")
    original_language = str(original_language).upper() if original_language and not pd.isna(original_language) else None
    status = row.get("status")
    status = str(status).strip() if status and not pd.isna(status) else None
    runtime_group = runtime_band(row.get("runtime"))

    return {
        "genres": unique_terms(genres[:5]),
        "keywords": unique_terms(keywords[:10]),
        "entities": unique_terms(entities[:8]),
        "companies": unique_terms(companies[:4]),
        "countries": unique_terms(countries[:3]),
        "languages": unique_terms(languages[:3]),
        "original_language": [original_language] if original_language else [],
        "decade": [decade] if decade else [],
        "runtime_band": [runtime_group] if runtime_group else [],
        "status": [status] if status else [],
    }


def build_movie_payload(row, score=0.0, matched_facts=None, score_breakdown=None):
    return {
        "title": row["title"],
        "overview": row["overview"],
        "rating": float(row["vote_average"]),
        "genre_list": parse_json_names(row["genres"]),
        "score": round(score, 4),
        "graph_score": 0.0,
        "transformer_score": 0.0,
        "matched_facts": matched_facts or [],
        "score_breakdown": score_breakdown or {},
    }


def build_transformer_text(row, concepts):
    parts = [
        row.get("title", ""),
        row.get("overview", ""),
        "Genres: " + ", ".join(concepts["genres"]),
        "Keywords: " + ", ".join(concepts["keywords"]),
        "Entities: " + ", ".join(concepts["entities"]),
        "Companies: " + ", ".join(concepts["companies"]),
        "Countries: " + ", ".join(concepts["countries"]),
        "Languages: " + ", ".join(concepts["languages"]),
        "Original language: " + ", ".join(concepts["original_language"]),
        "Decade: " + ", ".join(concepts["decade"]),
        "Runtime: " + ", ".join(concepts["runtime_band"]),
        "Status: " + ", ".join(concepts["status"]),
    ]
    return " ".join(part for part in parts if part and not part.endswith(": "))


# --- LOAD DATA ---
print(f"--- Initializing Knowledge Graph Engine (Limit: {LIMIT}) ---")
df = pd.read_csv(DATA_PATH)
df = df.head(LIMIT).reset_index(drop=True)
df["overview"] = df["overview"].fillna("")

movie_concepts = {}
relation_index = {relation_type: {} for relation_type in RELATION_WEIGHTS}

for idx, row in df.iterrows():
    concepts = build_movie_concepts(row)
    movie_concepts[idx] = concepts
    for relation_type, values in concepts.items():
        bucket = relation_index[relation_type]
        for value in values:
            bucket.setdefault(value.casefold(), {"value": value, "movies": set()})
            bucket[value.casefold()]["movies"].add(idx)

transformer_texts = [build_transformer_text(df.iloc[idx], movie_concepts[idx]) for idx in range(len(df))]

if SentenceTransformer is not None and cosine_similarity is not None:
    try:
        transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
        if os.path.exists(EMBEDDING_CACHE_PATH):
            transformer_embeddings = np.load(EMBEDDING_CACHE_PATH)
            if transformer_embeddings.shape[0] != len(df):
                transformer_embeddings = None
        if transformer_embeddings is None:
            transformer_embeddings = transformer_model.encode(transformer_texts, show_progress_bar=True)
            np.save(EMBEDDING_CACHE_PATH, transformer_embeddings)
        print("--- Transformer reranker enabled ---")
    except Exception:
        transformer_model = None
        transformer_embeddings = None
        print("--- Transformer reranker unavailable, continuing with graph-only mode ---")
else:
    print("--- sentence-transformers not installed, continuing with graph-only mode ---")


def relation_specificity(relation_type, value):
    bucket = relation_index[relation_type].get(value.casefold())
    if not bucket:
        return 1.0
    frequency = len(bucket["movies"])
    return 1.0 / math.log(frequency + 1.8)


def recommend_from_graph(target_idx, top_k=TOP_K):
    target_concepts = movie_concepts[target_idx]
    candidate_scores = {}
    candidate_matches = {}
    candidate_breakdown = {}

    for relation_type, values in target_concepts.items():
        relation_weight = RELATION_WEIGHTS[relation_type]
        for value in values:
            bucket = relation_index[relation_type].get(value.casefold())
            if not bucket:
                continue

            specificity = relation_specificity(relation_type, value)
            contribution = relation_weight * specificity

            for movie_idx in bucket["movies"]:
                if movie_idx == target_idx:
                    continue

                candidate_scores[movie_idx] = candidate_scores.get(movie_idx, 0.0) + contribution
                candidate_matches.setdefault(movie_idx, []).append({
                    "type": relation_type,
                    "label": DISPLAY_LABELS[relation_type],
                    "value": bucket["value"],
                    "weight": round(contribution, 4),
                })

                breakdown = candidate_breakdown.setdefault(movie_idx, {})
                breakdown[relation_type] = round(breakdown.get(relation_type, 0.0) + contribution, 4)

    transformer_bonus = {}
    if transformer_embeddings is not None and cosine_similarity is not None:
        similarity_scores = cosine_similarity([transformer_embeddings[target_idx]], transformer_embeddings)[0]
        for movie_idx in candidate_scores:
            transformer_bonus[movie_idx] = max(0.0, float(similarity_scores[movie_idx]))

    ranked = sorted(
        candidate_scores.items(),
        key=lambda item: (
            (item[1] * HYBRID_GRAPH_WEIGHT) + (transformer_bonus.get(item[0], 0.0) * HYBRID_TRANSFORMER_WEIGHT * 10.0),
            item[1],
            df.iloc[item[0]]["vote_average"],
            df.iloc[item[0]]["popularity"]
        ),
        reverse=True
    )[:top_k]

    recommendations = []
    for movie_idx, score in ranked:
        matches = sorted(candidate_matches.get(movie_idx, []), key=lambda item: item["weight"], reverse=True)
        breakdown = candidate_breakdown.get(movie_idx, {})
        graph_score = round(score, 4)
        transformer_score = round(transformer_bonus.get(movie_idx, 0.0), 4)
        hybrid_score = round((graph_score * HYBRID_GRAPH_WEIGHT) + (transformer_score * HYBRID_TRANSFORMER_WEIGHT * 10.0), 4)
        recommendations.append({
            "index": int(movie_idx),
            "score": hybrid_score,
            "graph_score": graph_score,
            "transformer_score": transformer_score,
            "matched_facts": matches,
            "score_breakdown": breakdown,
        })

    return recommendations


def build_graph_payload(target_idx, recommendation_bundle):
    target_row = df.iloc[target_idx]
    target_title = target_row["title"]
    target_movie_id = f"movie::{target_title}"
    target_concepts = movie_concepts[target_idx]

    nodes = [{
        "id": target_movie_id,
        "label": target_title,
        "type": "movie",
        "role": "source",
        "score": 1.0,
    }]
    edges = []
    concept_nodes = {}

    def ensure_concept_node(match):
        node_id = relation_id(match["type"], match["value"])
        if node_id not in concept_nodes:
            concept_nodes[node_id] = {
                "id": node_id,
                "label": match["value"],
                "type": match["type"],
                "role": "concept",
                "score": 0.0,
            }
        concept_nodes[node_id]["score"] += match["weight"]
        return node_id

    for rec in recommendation_bundle:
        movie_idx = rec["index"]
        movie_row = df.iloc[movie_idx]
        movie_id = f"movie::{movie_row['title']}"

        nodes.append({
            "id": movie_id,
            "label": movie_row["title"],
            "type": "movie",
            "role": "recommendation",
            "score": rec["score"],
        })
        edges.append({
            "source": target_movie_id,
            "target": movie_id,
            "type": "hybrid-score",
            "weight": rec["score"],
            "label": f"{rec['score']:.3f}",
        })

        for match in rec["matched_facts"][:8]:
            node_id = ensure_concept_node(match)
            if not any(edge["source"] == target_movie_id and edge["target"] == node_id for edge in edges):
                edges.append({
                    "source": target_movie_id,
                    "target": node_id,
                    "type": "has-fact",
                    "weight": match["weight"],
                    "label": match["label"],
                })
            edges.append({
                "source": node_id,
                "target": movie_id,
                "type": "shares-fact",
                "weight": match["weight"],
                "label": match["label"],
            })

    nodes.extend(sorted(concept_nodes.values(), key=lambda item: (-item["score"], item["label"])))

    return {
        "nodes": nodes,
        "edges": edges,
        "knowledge_profile": target_concepts,
        "scoring": {
            "method": "weighted graph traversal + transformer reranking" if transformer_embeddings is not None else "weighted graph traversal",
            "relations": RELATION_WEIGHTS,
            "hybrid_weights": {
                "graph": HYBRID_GRAPH_WEIGHT,
                "transformer": HYBRID_TRANSFORMER_WEIGHT,
            },
            "transformer_enabled": transformer_embeddings is not None,
        }
    }


# --- API ENDPOINTS ---
@app.get("/search")
def search_titles(query: str):
    if not query:
        return []
    matches = df[df["title"].str.contains(query, case=False, na=False)]["title"].head(6).tolist()
    return matches


@app.get("/recommend")
def get_recs(title: str):
    titles = df["title"].tolist()
    match = difflib.get_close_matches(title, titles, n=1, cutoff=0.5)

    if not match:
        return {"error": "Movie not found in current dataset subset."}

    target_title = match[0]
    target_idx = df[df["title"] == target_title].index[0]
    recommendation_bundle = recommend_from_graph(target_idx, TOP_K)

    recommendations = []
    for rec in recommendation_bundle:
        row = df.iloc[rec["index"]]
        recommendations.append(build_movie_payload(
            row,
            score=rec["score"],
            matched_facts=rec["matched_facts"][:8],
            score_breakdown=rec["score_breakdown"],
        ))
        recommendations[-1]["graph_score"] = rec["graph_score"]
        recommendations[-1]["transformer_score"] = rec["transformer_score"]

    graph = build_graph_payload(target_idx, recommendation_bundle)

    return {
        "searched_for": target_title,
        "engine": "knowledge-graph",
        "data": recommendations,
        "graph": graph,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
