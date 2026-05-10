import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from pdf_processor import process_pdf
from vector_store import VectorStore

load_dotenv()

st.set_page_config(
    page_title="Медицинская библиотека",
    page_icon="📚",
    layout="wide",
)

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


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore()


# ---------------------------------------------------------------------------
# Sidebar — upload & book list
# ---------------------------------------------------------------------------

store = get_store()

with st.sidebar:
    st.title("📚 Учебники")

    uploaded_files = st.file_uploader(
        "Загрузить PDF",
        type="pdf",
        accept_multiple_files=True,
        help="Файлы любого размера",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if store.book_exists(uploaded_file.name):
                st.info(f"«{uploaded_file.name}» уже загружена")
                continue

            with st.status(f"Обрабатываю «{uploaded_file.name}»...", expanded=True) as status:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    st.write("Фаза 1: извлекаю текст...")
                    text_bar = st.progress(0)

                    def text_cb(current, total):
                        text_bar.progress(min(current / total, 1.0))

                    ocr_label = st.empty()
                    ocr_bar = st.progress(0)

                    def ocr_cb(current, total):
                        ocr_label.write(f"Фаза 2: OCR страница {current}/{total}...")
                        ocr_bar.progress(min(current / total, 1.0))

                    chunks = process_pdf(
                        tmp_path, text_cb, ocr_cb,
                        original_filename=uploaded_file.name,
                    )
                    text_bar.progress(1.0)

                    if not chunks:
                        status.update(label="Ошибка", state="error")
                        st.error(
                            "Не удалось извлечь текст. "
                            "Возможно, PDF содержит только сканированные изображения."
                        )
                    else:
                        st.write(f"Индексирую {len(chunks)} фрагментов...")
                        store.add_chunks(chunks)
                        first = chunks[0]
                        status.update(
                            label=f"✅ «{first['title']}» добавлена",
                            state="complete",
                        )
                except Exception as e:
                    status.update(label="Ошибка", state="error")
                    st.error(str(e))
                finally:
                    os.unlink(tmp_path)

    # Book list
    books = store.list_books()
    if books:
        st.divider()
        st.subheader("Загружено")
        for b in books:
            st.markdown(
                f"📖 **{b['title']}**  \n"
                f"_{b['author']}_ · {b['chunk_count']} фрагм."
            )

        st.divider()
        if st.button("🗑️ Удалить все книги", use_container_width=True):
            store.delete_all()
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("Загрузи PDF-учебник чтобы начать")

# ---------------------------------------------------------------------------
# Main area — chat
# ---------------------------------------------------------------------------

st.title("Медицинская библиотека")
st.caption("Отвечаю только по загруженным учебникам · Все ответы со ссылками на страницы")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Источники"):
                for i, chunk in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**[{i}] {chunk['author']}, «{chunk['title']}», стр. {chunk['page']}**"
                    )
                    st.caption(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))

# Input
if question := st.chat_input("Задай вопрос по учебникам..."):
    if not store.list_books():
        st.warning("Сначала загрузи учебник — используй панель слева.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            top_k = int(os.getenv("TOP_K_CHUNKS", 20))

            # Expand query: generate alternative phrasings to improve recall
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
                        f"Вопрос: {question}"
                    )
                }],
                temperature=0.3,
                max_tokens=200,
            )
            expansions = expansion_response.choices[0].message.content.strip().split("\n")
            queries = [question] + [q.strip() for q in expansions if q.strip()]

            # Search for all query variants and deduplicate by chunk_id
            seen_ids = set()
            all_chunks = []
            for q in queries:
                for chunk in store.search(q, top_k=top_k // len(queries) + 2):
                    cid = chunk.get("filename", "") + str(chunk.get("page", "")) + chunk["text"][:50]
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        all_chunks.append(chunk)

            # Sort by distance (best matches first) and take top_k
            all_chunks.sort(key=lambda c: c.get("distance", 1.0))
            chunks = all_chunks[:top_k]
            context_str = _build_context(chunks)

            # Build messages with history (last 3 exchanges)
            history = [
                m for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
            ][-6:]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context_str)},
                *[{"role": m["role"], "content": m["content"]} for m in history],
                {"role": "user", "content": question},
            ]

            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
                temperature=0.1,
                max_tokens=16000,
            )
            answer = response.choices[0].message.content

        st.markdown(answer)
        with st.expander("📚 Источники"):
            for i, chunk in enumerate(chunks, 1):
                st.markdown(
                    f"**[{i}] {chunk['author']}, «{chunk['title']}», стр. {chunk['page']}**"
                )
                st.caption(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks,
    })
