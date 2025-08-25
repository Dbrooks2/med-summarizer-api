from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import re

app = FastAPI(title="Medical Report Summarization API", version="1.0.0")

# Data models
class SummarizeRequest(BaseModel):
    text: str
    max_words: Optional[int] = 100

class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class RAGSummarizeRequest(BaseModel):
    query: str
    k: Optional[int] = 3
    max_words: Optional[int] = 100

class BuildIndexRequest(BaseModel):
    csv_path: str
    text_column: Optional[str] = "transcription"

# Global variables for loaded models
tfidf_vectorizer = None
retrieval_matrix = None
retrieval_corpus = None

def load_artifacts():
    """Load the pre-built artifacts if they exist"""
    global tfidf_vectorizer, retrieval_matrix, retrieval_corpus
    
    data_dir = "/mnt/data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return False
    
    try:
        tfidf_path = os.path.join(data_dir, "retrieval_tfidf.joblib")
        matrix_path = os.path.join(data_dir, "retrieval_matrix.npz")
        corpus_path = os.path.join(data_dir, "retrieval_corpus.csv")
        
        if all(os.path.exists(p) for p in [tfidf_path, matrix_path, corpus_path]):
            tfidf_vectorizer = joblib.load(tfidf_path)
            retrieval_matrix = np.load(matrix_path)['arr_0']
            retrieval_corpus = pd.read_csv(corpus_path)
            return True
    except Exception as e:
        print(f"Error loading artifacts: {e}")
    
    return False

def extractive_summarize(text: str, max_words: int = 100) -> str:
    """Simple extractive summarization using TF-IDF scoring"""
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return text
    
    # Simple word frequency scoring
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = {}
    for word in words:
        if len(word) > 2:  # Skip very short words
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Score sentences
    sentence_scores = []
    for sentence in sentences:
        score = sum(word_freq.get(word.lower(), 0) for word in re.findall(r'\b\w+\b', sentence))
        sentence_scores.append((score, sentence))
    
    # Sort by score and take top sentences
    sentence_scores.sort(reverse=True)
    
    summary = []
    word_count = 0
    for _, sentence in sentence_scores:
        sentence_words = len(re.findall(r'\b\w+\b', sentence))
        if word_count + sentence_words <= max_words:
            summary.append(sentence)
            word_count += sentence_words
        else:
            break
    
    return '. '.join(summary) + ('.' if summary else '')

@app.on_event("startup")
async def startup_event():
    """Load artifacts on startup"""
    load_artifacts()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "artifacts_loaded": tfidf_vectorizer is not None}

@app.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    """Summarize medical text using extractive summarization"""
    try:
        summary = extractive_summarize(request.text, request.max_words)
        return {
            "summary": summary,
            "original_length": len(request.text.split()),
            "summary_length": len(summary.split()),
            "max_words": request.max_words
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")

@app.post("/retrieve")
async def retrieve_documents(request: RetrieveRequest):
    """Retrieve relevant documents using TF-IDF similarity"""
    if tfidf_vectorizer is None:
        raise HTTPException(status_code=400, detail="No retrieval index built. Use /build_index first.")
    
    try:
        # Transform query
        query_vector = tfidf_vectorizer.transform([request.query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, retrieval_matrix).flatten()
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:request.top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    "index": int(idx),
                    "similarity": float(similarities[idx]),
                    "text": retrieval_corpus.iloc[idx].iloc[0]  # First column should be text
                })
        
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

@app.post("/rag_summarize")
async def rag_summarize(request: RAGSummarizeRequest):
    """RAG-based summarization: retrieve relevant docs then summarize"""
    if tfidf_vectorizer is None:
        raise HTTPException(status_code=400, detail="No retrieval index built. Use /build_index first.")
    
    try:
        # First retrieve relevant documents
        query_vector = tfidf_vectorizer.transform([request.query])
        similarities = cosine_similarity(query_vector, retrieval_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:request.k]
        
        # Combine relevant texts
        relevant_texts = []
        for idx in top_indices:
            if similarities[idx] > 0:
                relevant_texts.append(retrieval_corpus.iloc[idx].iloc[0])
        
        if not relevant_texts:
            return {"summary": "No relevant documents found.", "query": request.query}
        
        # Combine and summarize
        combined_text = " ".join(relevant_texts)
        summary = extractive_summarize(combined_text, request.max_words)
        
        return {
            "summary": summary,
            "query": request.query,
            "documents_retrieved": len(relevant_texts),
            "summary_length": len(summary.split()),
            "max_words": request.max_words
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG summarization error: {str(e)}")

@app.post("/build_index")
async def build_index(request: BuildIndexRequest):
    """Build the retrieval index from a CSV file"""
    try:
        # Check if CSV exists
        if not os.path.exists(request.csv_path):
            raise HTTPException(status_code=400, detail=f"CSV file not found: {request.csv_path}")
        
        # Run the build script directly
        import subprocess
        import sys
        
        # Create data directory if it doesn't exist
        os.makedirs("/mnt/data", exist_ok=True)
        
        # Run the build script
        result = subprocess.run([
            sys.executable, 
            "scripts/build_index.py",
            "--csv", request.csv_path,
            "--text-col", request.text_column,
            "--out-dir", "/mnt/data"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Build script failed: {result.stderr}")
        
        # Reload artifacts
        load_artifacts()
        
        return {
            "message": "Index built successfully",
            "csv_path": request.csv_path,
            "text_column": request.text_column,
            "artifacts_loaded": tfidf_vectorizer is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index building error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
