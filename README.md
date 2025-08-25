# Medical Report Summarization API (FastAPI)

AI-powered extractive summarization and retrieval for clinical notes.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Artifacts expected (or build via `/build_index` endpoint):
- `/mnt/data/retrieval_tfidf.joblib`
- `/mnt/data/retrieval_matrix.npz`
- `/mnt/data/retrieval_corpus.csv`

Mount a local `data/` directory to `/mnt/data` (see docker-compose).

## Docker

```bash
docker build -t med-summarizer-api .
docker run -p 8000:8000 -v $(pwd)/data:/mnt/data med-summarizer-api
```

Or with compose:

```bash
docker compose up --build
```

## Endpoints
- `GET /health`
- `POST /summarize`  -> {text, max_words?}
- `POST /retrieve`   -> {query, top_k?}
- `POST /rag_summarize` -> {query, k?, max_words?}
- `POST /build_index` -> {csv_path, text_column?}

## Push to GitHub

1) Create a new empty repo on GitHub (e.g., `med-summarizer-api`).
2) In a terminal:

```bash
git init
git add .
git commit -m "Initial commit: medical summarization FastAPI"
git branch -M main
git remote add origin https://github.com/<your-username>/med-summarizer-api.git
# or: git@github.com:<your-username>/med-summarizer-api.git
git push -u origin main
```

> Note: This project is **not a medical device** and does not provide medical advice.
