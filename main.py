import asyncio
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from pdf_processor import process_pdf
from vector_store import VectorStore

load_dotenv()

app = FastAPI(title="Медицинская библиотека")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

store = VectorStore()
executor = ThreadPoolExecutor(max_workers=2)

SYSTEM_PROMPT = """Ты — учебный ассистент по медицине. Твоя единственная задача — отвечать на вопросы, \
используя ТОЛЬКО текст из фрагментов учебников, которые тебе предоставлены ниже.

СТРОГИЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленных фрагментов. Не используй никакие внешние знания.
2. Если в предоставленных фрагментах нет информации для ответа — скажи точно: \
"В загруженных учебниках нет информации по этому вопросу."
3. После каждого факта или утверждения указывай источник: [Автор, «Название», стр. X]
4. Не выдумывай и не дополняй информацию за пределами предоставленного текста.
5. Если разные источники противоречат друг другу — укажи оба и отметь противоречие.
6. Отвечай на том же языке, на котором задан вопрос.
7. Копируй текст из фрагментов ДОСЛОВНО и ПОЛНОСТЬЮ — не сокращай, не перефразируй, \
не суммируй. Просто выдай весь релевантный текст из источников как есть, с цитатами.

ФРАГМЕНТЫ ИЗ УЧЕБНИКОВ:
{context}
"""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Автор: {c['author']} | Книга: {c['title']} | Стр. {c['page']}\n"
            f"---\n"
            f"{c['text']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Только PDF файлы")

    if store.book_exists(file.filename):
        return {"status": "exists", "filename": file.filename}

    content = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            executor,
            lambda: process_pdf(tmp_path, None, None, original_filename=file.filename),
        )
        if not chunks:
            raise HTTPException(status_code=422, detail="Не удалось извлечь текст из PDF")
        store.add_chunks(chunks)
        return {
            "status": "ok",
            "title": chunks[0]["title"],
            "author": chunks[0]["author"],
            "chunks": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.get("/api/books")
def list_books():
    return store.list_books()


@app.delete("/api/books")
def delete_books():
    store.delete_all()
    return {"status": "ok"}


class AskRequest(BaseModel):
    question: str
    history: list[dict] = []


@app.post("/api/ask")
def ask_question(body: AskRequest):
    def generate():
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        top_k = int(os.getenv("TOP_K_CHUNKS", 20))

        # Query expansion
        try:
            expansion_response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{
                    "role": "user",
                    "content": (
                        f"Ты помогаешь искать информацию в медицинских учебниках на русском и английском языках.\n"
                        f"Сделай следующее для вопроса ниже:\n"
                        f"1. Исправь опечатки если есть\n"
                        f"2. Напиши 2 варианта на русском (синонимы, медицинские термины)\n"
                        f"3. Напиши 2 варианта на английском (перевод + медицинские термины)\n"
                        f"Верни только список из 4 вариантов, каждый на новой строке, без нумерации и пояснений.\n\n"
                        f"Вопрос: {body.question}"
                    ),
                }],
                temperature=0.3,
                max_tokens=200,
            )
            expansions = expansion_response.choices[0].message.content.strip().split("\n")
            queries = [body.question] + [q.strip() for q in expansions if q.strip()]
        except Exception:
            queries = [body.question]

        # Search
        seen_ids: set[str] = set()
        all_chunks: list[dict] = []
        for q in queries:
            for chunk in store.search(q, top_k=top_k // len(queries) + 2):
                cid = chunk.get("filename", "") + str(chunk.get("page", "")) + chunk["text"][:50]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(chunk)

        all_chunks.sort(key=lambda c: c.get("distance", 1.0))
        chunks = all_chunks[:top_k]
        context_str = _build_context(chunks)

        history = [m for m in body.history if m.get("role") in ("user", "assistant")][-6:]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context_str)},
            *[{"role": m["role"], "content": m["content"]} for m in history],
            {"role": "user", "content": body.question},
        ]

        # Stream answer
        stream = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.1,
            max_tokens=16000,
            stream=True,
        )

        for event in stream:
            delta = event.choices[0].delta
            if delta.content:
                yield f"data: {json.dumps({'type': 'token', 'text': delta.content})}\n\n"

        # Send sources
        sources = [
            {"author": c["author"], "title": c["title"], "page": c["page"], "text": c["text"][:300]}
            for c in chunks
        ]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Static files (must be last)
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")
