import asyncio
import logging
import os
import shutil
from functools import partial

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from pdf_processor import process_pdf
from vector_store import VectorStore

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — the core of the "answer only from books" behaviour
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


def _split_message(text: str, limit: int = 4096) -> list[str]:
    """Split long text into Telegram-safe chunks."""
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        parts.append(text[:limit])
        text = text[limit:]
    return parts


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я ассистент по медицинским учебникам.\n\n"
        "📚 Отправь мне PDF-файл учебника — я его изучу.\n"
        "❓ Затем задавай любые вопросы. Отвечу только на основе загруженных книг "
        "и укажу автора и страницу.\n\n"
        "Команды:\n"
        "/books — список загруженных учебников\n"
        "/clear — очистить историю диалога\n"
        "/reset — удалить все учебники и начать заново"
    )


async def cmd_books(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store: VectorStore = context.bot_data["store"]
    books = store.list_books()
    if not books:
        await update.message.reply_text("Учебники ещё не загружены. Отправь PDF-файл.")
        return
    lines = ["📚 Загруженные учебники:\n"]
    for b in books:
        lines.append(
            f"• {b['title']}\n"
            f"  Автор: {b['author']}\n"
            f"  Файл: {b['filename']}\n"
            f"  Фрагментов в базе: {b['chunk_count']}"
        )
    await update.message.reply_text("\n\n".join(lines))


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["history"] = []
    await update.message.reply_text("История диалога очищена.")


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store: VectorStore = context.bot_data["store"]
    store.delete_all()
    if os.path.exists("books"):
        shutil.rmtree("books")
    os.makedirs("books", exist_ok=True)
    context.user_data["history"] = []
    await update.message.reply_text(
        "Все учебники удалены. Можешь загрузить новые — просто отправь PDF."
    )


# ---------------------------------------------------------------------------
# PDF upload handler
# ---------------------------------------------------------------------------


async def handle_pdf_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store: VectorStore = context.bot_data["store"]
    doc = update.message.document
    filename = doc.file_name or "book.pdf"

    # Duplicate check
    if store.book_exists(filename):
        await update.message.reply_text(
            f"Книга «{filename}» уже загружена.\n"
            "Используй /reset, чтобы удалить все книги и начать заново."
        )
        return

    # Size check (Telegram bots can only download files up to ~20 MB)
    if doc.file_size and doc.file_size > 20 * 1024 * 1024:
        await update.message.reply_text(
            "Файл слишком большой — Telegram позволяет ботам принимать максимум 20 МБ.\n\n"
            "Что можно сделать:\n"
            "• Сжать PDF на ilovepdf.com/compress_pdf\n"
            "• Разбить на части на ilovepdf.com/split_pdf"
        )
        return

    status_msg = await update.message.reply_text(f"Получил «{filename}». Обрабатываю... ⏳")

    # Download
    os.makedirs("books", exist_ok=True)
    file_path = os.path.join("books", filename)
    tg_file = await context.bot.get_file(doc.file_id)
    await tg_file.download_to_drive(file_path)

    try:
        # Progress callback — edit the status message from the sync thread
        loop = asyncio.get_event_loop()

        def progress_cb(current, total):
            asyncio.run_coroutine_threadsafe(
                status_msg.edit_text(
                    f"Извлекаю текст: страница {current}/{total}... ⏳"
                ),
                loop,
            )

        # Run pdfplumber in thread pool so it doesn't block Telegram event loop
        chunks = await loop.run_in_executor(
            None, partial(process_pdf, file_path, progress_cb)
        )

        if not chunks:
            await status_msg.edit_text(
                "Не удалось извлечь текст из файла.\n"
                "Возможно, PDF содержит только сканированные изображения (OCR не поддерживается)."
            )
            return

        await status_msg.edit_text(f"Индексирую {len(chunks)} фрагментов... ⏳")

        # Embedding and storing (can be slow for large books)
        await loop.run_in_executor(None, store.add_chunks, chunks)

        first = chunks[0]
        last_page = chunks[-1]["page"]
        await status_msg.edit_text(
            f"Готово! Книга добавлена:\n"
            f"📖 Название: {first['title']}\n"
            f"✍️ Автор: {first['author']}\n"
            f"📄 Страниц обработано: {last_page}\n"
            f"🔍 Фрагментов в базе: {len(chunks)}\n\n"
            f"Задавай вопросы!"
        )

    except Exception as e:
        logger.exception("PDF processing error")
        await status_msg.edit_text(
            f"Ошибка при обработке файла:\n{e}\n\n"
            "Попробуй другой файл или обратись к разработчику."
        )


# ---------------------------------------------------------------------------
# Question handler
# ---------------------------------------------------------------------------


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store: VectorStore = context.bot_data["store"]
    question = update.message.text.strip()

    if not store.list_books():
        await update.message.reply_text(
            "Сначала загрузи учебник — отправь PDF-файл в этот чат."
        )
        return

    thinking_msg = await update.message.reply_text("Думаю... ⏳")

    try:
        top_k = int(os.getenv("TOP_K_CHUNKS", 10))
        chunks = store.search(question, top_k=top_k)
        context_str = _build_context(chunks)

        history = context.user_data.get("history", [])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context_str)},
            *history[-6:],  # last 3 exchanges
            {"role": "user", "content": question},
        ]

        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.1,
            max_tokens=1500,
        )
        answer = response.choices[0].message.content

        # Save to history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        context.user_data["history"] = history

        # Send (split if too long)
        parts = _split_message(answer)
        await thinking_msg.edit_text(parts[0])
        for part in parts[1:]:
            await update.message.reply_text(part)

    except Exception as e:
        logger.exception("Question handling error")
        await thinking_msg.edit_text(
            f"Произошла ошибка: {e}\n"
            "Попробуй ещё раз или перезапусти бота."
        )


# ---------------------------------------------------------------------------
# Catch-all for non-PDF documents
# ---------------------------------------------------------------------------


async def handle_non_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Поддерживаются только PDF-файлы. Отправь учебник в формате .pdf"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан. Проверь файл .env")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY не задан. Проверь файл .env")

    store = VectorStore()

    app = ApplicationBuilder().token(token).build()
    app.bot_data["store"] = store

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("books", cmd_books))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("reset", cmd_reset))

    app.add_handler(
        MessageHandler(filters.Document.MimeType("application/pdf"), handle_pdf_upload)
    )
    app.add_handler(
        MessageHandler(filters.Document.ALL & ~filters.Document.MimeType("application/pdf"), handle_non_pdf)
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    logger.info("Bot started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
