# PDF Question Answering Chatbot

A beginner-friendly Retrieval-Augmented Generation (RAG) application. Upload a PDF, retrieve relevant passages with TF-IDF, and ask questions. An optional OpenAI integration generates answers from the retrieved passages. Without an API key, the app returns the most relevant excerpts.

## Skills demonstrated
Python, FastAPI, PDF text extraction, text chunking, retrieval, embeddings concept (TF-IDF baseline), REST APIs, environment variables, basic testing.

> This starter uses TF-IDF retrieval so it runs without a paid vector database. It is an educational RAG-style baseline, not a production vector-search system.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs

Set `OPENAI_API_KEY` in `.env` to enable generated answers. Never commit your real API key.

## API
- `POST /documents` — upload a PDF (`file` form field)
- `POST /ask` — JSON body: `{"question": "..."}`

## Before publishing
Run the app, test with a PDF you are allowed to use, add screenshots, and explain chunking, retrieval, and the limits of the approach.
