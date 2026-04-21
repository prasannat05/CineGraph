import pandas as pd
import numpy as np
import spacy
import json
import os
import difflib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- INITIALIZATION ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Configuration
DATA_PATH = "tmdb_5000.csv"
LIMIT = 3000  # Change this number safely now
CACHE_PATH = f"movie_vectors_{LIMIT}.npy" 

try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

transformer = SentenceTransformer('all-MiniLM-L6-v2')

# --- DATA PROCESSING HELPERS ---
def safe_parse_json(data):
    try:
        if pd.isna(data): return ""
        items = json.loads(data)
        return " ".join([d['name'] for d in items])
    except:
        return ""

def extract_ner_nodes(text):
    if not text or len(text) < 10: return ""
    doc = nlp(str(text))
    entities = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'PERSON', 'ORG']]
    return " ".join(entities)

# --- LOAD AND PREPARE DATA ---
print(f"--- Initializing Engine (Limit: {LIMIT}) ---")
df = pd.read_csv(DATA_PATH)
df = df.head(LIMIT).reset_index(drop=True) # Reset index to match matrix rows exactly

df['genres_clean'] = df['genres'].apply(safe_parse_json)
df['keywords_clean'] = df['keywords'].apply(safe_parse_json)
df['overview'] = df['overview'].fillna('')

if os.path.exists(CACHE_PATH):
    print(f"--- Loading Cache: {CACHE_PATH} ---")
    embeddings = np.load(CACHE_PATH)
    # Double check if cache actually matches current df size
    if embeddings.shape[0] != len(df):
        print("--- Cache size mismatch! Regenerating... ---")
        os.remove(CACHE_PATH)
        # Restarting the logic below...
else:
    print("--- Building Knowledge Soup & Vectors ---")
    df['ner_nodes'] = df['overview'].apply(extract_ner_nodes)
    df['knowledge_soup'] = (
        df['overview'] + " " + 
        df['genres_clean'] + " " + 
        df['keywords_clean'] + " " + 
        df['ner_nodes']
    ).astype(str)
    
    embeddings = transformer.encode(df['knowledge_soup'].tolist(), show_progress_bar=True)
    np.save(CACHE_PATH, embeddings)

# --- API ENDPOINTS ---
@app.get("/search")
def search_titles(query: str):
    if not query: return []
    matches = df[df['title'].str.contains(query, case=False, na=False)]['title'].head(6).tolist()
    return matches

@app.get("/recommend")
def get_recs(title: str):
    titles = df['title'].tolist()
    match = difflib.get_close_matches(title, titles, n=1, cutoff=0.5)
    
    if not match:
        return {"error": "Movie not found in current dataset subset."}
    
    target_title = match[0]
    # Get the row index that corresponds 1:1 with the embedding matrix
    idx = df[df['title'] == target_title].index[0]
    
    # Cosine Similarity
    cosine_scores = cosine_similarity([embeddings[idx]], embeddings)[0]
    top_indices = cosine_scores.argsort()[-7:-1][::-1] 
    
    recommendations = []
    for i in top_indices:
        recommendations.append({
            "title": df.iloc[i]['title'],
            "overview": df.iloc[i]['overview'],
            "rating": float(df.iloc[i]['vote_average']),
            "genres": df.iloc[i]['genres_clean'],
            "keywords": df.iloc[i]['keywords_clean']
        })
        
    return {"searched_for": target_title, "data": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)