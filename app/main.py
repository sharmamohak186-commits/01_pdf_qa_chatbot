import os
from io import BytesIO
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
app = FastAPI(title="PDF Question Answering Chatbot", version="1.0.0")

# In-memory storage keeps this starter simple. Documents are cleared on restart.
chunks: List[str] = []
vectorizer = None
chunk_matrix = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def split_into_chunks(text: str, words_per_chunk: int = 180, overlap: int = 35) -> List[str]:
    words = text.split()
    if not words:
        return []
    result = []
    step = max(1, words_per_chunk - overlap)
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + words_per_chunk])
        if piece:
            result.append(piece)
    return result


@app.get("/")
def health():
    return {"status": "ok", "message": "Upload a PDF, then ask a question at /docs."}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    global chunks, vectorizer, chunk_matrix
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF must be smaller than 10 MB.")
    try:
        reader = PdfReader(BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}")
    new_chunks = split_into_chunks(text)
    if not new_chunks:
        raise HTTPException(status_code=422, detail="No extractable text found. Scanned PDFs need OCR.")
    chunks = new_chunks
    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_matrix = vectorizer.fit_transform(chunks)
    return {"filename": file.filename, "pages": len(reader.pages), "chunks_indexed": len(chunks)}


def retrieve(question: str, top_k: int = 3) -> List[str]:
    if vectorizer is None or chunk_matrix is None:
        raise HTTPException(status_code=400, detail="Upload a PDF before asking questions.")
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, chunk_matrix).flatten()
    indices = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in indices if scores[i] > 0]


def generate_answer(question: str, context: List[str]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "No OpenAI API key configured. Relevant excerpts:\n\n" + "\n\n---\n\n".join(context)
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "Answer only using the supplied document context. If the answer is not present, say you cannot find it in the document. Do not invent facts."},
            {"role": "user", "content": f"Context:\n{chr(10).join(context)}\n\nQuestion: {question}"}
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or "No answer generated."


@app.post("/ask")
def ask(request: AskRequest):
    context = retrieve(request.question)
    if not context:
        return {"answer": "I could not find a relevant passage in the uploaded PDF.", "sources": []}
    return {"answer": generate_answer(request.question, context), "sources": context}
