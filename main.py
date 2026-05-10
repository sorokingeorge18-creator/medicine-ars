import asyncio
import base64
import hashlib
import hmac
import json
import os
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request

from pdf_processor import process_pdf
from vector_store import VectorStore

load_dotenv()

app = FastAPI(title="Медицинская библиотека")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "dev-secret-change-me"))


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return base64.b64encode(salt + key).decode()


def _verify_password(password: str, stored: str) -> bool:
    data = base64.b64decode(stored.encode())
    salt, key = data[:16], data[16:]
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return hmac.compare_digest(key, check)

# ---------------------------------------------------------------------------
# SQLite user storage
# ---------------------------------------------------------------------------

DB_PATH = Path("./chroma_db/users.db")


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                email    TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                name     TEXT NOT NULL
            )
        """)


init_db()

# ---------------------------------------------------------------------------
# Per-user VectorStore cache
# ---------------------------------------------------------------------------

_stores: dict[str, VectorStore] = {}


def get_store(user_id: str) -> VectorStore:
    if user_id not in _stores:
        _stores[user_id] = VectorStore(user_id=user_id)
    return _stores[user_id]


executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def require_user(request: Request) -> dict:
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Необходима авторизация")
    return user


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

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
# Auth endpoints
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/register")
async def register(body: RegisterRequest, request: Request):
    email = body.email.strip().lower()
    name = body.name.strip()
    if not email or not body.password or not name:
        raise HTTPException(status_code=400, detail="Заполните все поля")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Пароль должен быть не менее 6 символов")
    hashed = _hash_password(body.password)
    try:
        with _db() as conn:
            cursor = conn.execute(
                "INSERT INTO users (email, password, name) VALUES (?, ?, ?)",
                (email, hashed, name),
            )
            user_id = str(cursor.lastrowid)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Этот email уже зарегистрирован")
    request.session["user"] = {"sub": user_id, "email": email, "name": name}
    return {"status": "ok", "name": name, "email": email}


@app.post("/auth/login")
async def login(body: LoginRequest, request: Request):
    email = body.email.strip().lower()
    with _db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not row or not _verify_password(body.password, row["password"]):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")
    request.session["user"] = {
        "sub": str(row["id"]),
        "email": row["email"],
        "name": row["name"],
    }
    return {"status": "ok", "name": row["name"], "email": row["email"]}


@app.get("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return {"status": "ok"}


@app.get("/auth/me")
async def get_me(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401)
    return user


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...), user: dict = Depends(require_user)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Только PDF файлы")

    store = get_store(user["sub"])
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
def list_books(user: dict = Depends(require_user)):
    return get_store(user["sub"]).list_books()


@app.delete("/api/books")
def delete_books(user: dict = Depends(require_user)):
    get_store(user["sub"]).delete_all()
    return {"status": "ok"}


@app.delete("/api/books/{filename:path}")
def delete_book(filename: str, user: dict = Depends(require_user)):
    get_store(user["sub"]).delete_book(filename)
    return {"status": "ok"}


class AskRequest(BaseModel):
    question: str
    history: list[dict] = []


@app.post("/api/ask")
def ask_question(body: AskRequest, request: Request, user: dict = Depends(require_user)):
    user_store = get_store(user["sub"])

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
            for chunk in user_store.search(q, top_k=top_k // len(queries) + 2):
                cid = chunk.get("filename", "") + str(chunk.get("page", "")) + chunk["text"][:50]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(chunk)

        all_chunks.sort(key=lambda c: c.get("distance", 1.0))
        chunks = all_chunks[:top_k]

        if not chunks:
            yield f"data: {json.dumps({'type': 'token', 'text': 'В библиотеке нет загруженных учебников. Перейдите в «Центр загрузки» и добавьте PDF-файлы.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        context_str = _build_context(chunks)

        history = [m for m in body.history if m.get("role") in ("user", "assistant")][-6:]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context_str)},
            *[{"role": m["role"], "content": m["content"]} for m in history],
            {"role": "user", "content": body.question},
        ]

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
