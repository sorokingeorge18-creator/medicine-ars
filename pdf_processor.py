import os
import re
from pathlib import Path


def _fix_char_spacing(text: str) -> str:
    """Fix PDFs where characters are stored individually: 'И л л ю с т р а' → 'Иллюстра'."""
    # Match 3+ single Cyrillic/Latin letters each separated by a single space
    pattern = r'(?<![А-Яа-яЁёA-Za-z])([А-Яа-яЁёA-Za-z] ){3,}[А-Яа-яЁёA-Za-z](?![А-Яа-яЁёA-Za-z])'
    def join_chars(m):
        return m.group(0).replace(' ', '')
    return re.sub(pattern, join_chars, text)


def _extract_author(metadata: dict, filename: str) -> str:
    for key in ("author", "Author", "creator", "Creator"):
        val = metadata.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    stem = Path(filename).stem
    parts = re.split(r"[_\-\s]+", stem)
    for part in parts:
        if len(part) >= 3 and part[0].isupper() and part.isalpha():
            return part
    return "Автор неизвестен"


def _extract_title(metadata: dict, filename: str) -> str:
    val = metadata.get("title") or metadata.get("Title")
    if val and isinstance(val, str) and val.strip():
        return val.strip()
    return Path(filename).stem.replace("_", " ")


def _chunk_text(text: str, max_chars: int = 3000, overlap_chars: int = 300) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            for sep in (".\n", ". ", "\n\n", "\n"):
                idx = text.rfind(sep, start + max_chars // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if len(chunk) > 20:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap_chars
    return chunks


def process_pdf(
    file_path: str,
    text_progress_cb=None,
    ocr_progress_cb=None,
    original_filename: str = None,
) -> list[dict]:
    import fitz  # pymupdf

    max_chars = int(os.getenv("CHUNK_SIZE", 800)) * 4
    overlap_chars = int(os.getenv("CHUNK_OVERLAP", 100)) * 4

    filename = original_filename or Path(file_path).name
    stem = Path(filename).stem
    all_chunks = []

    try:
        doc = fitz.open(file_path)
        metadata = doc.metadata or {}
        author = _extract_author(metadata, filename)
        title = _extract_title(metadata, filename)
        total_pages = len(doc)

        for i in range(total_pages):
            if text_progress_cb:
                text_progress_cb(i + 1, total_pages)
            try:
                page = doc[i]
                text = page.get_text("text") or ""
                text = _fix_char_spacing(text)
            except Exception:
                text = ""
            if not text or len(text.strip()) < 20:
                continue
            for j, chunk in enumerate(_chunk_text(text, max_chars, overlap_chars)):
                all_chunks.append({
                    "text": chunk,
                    "page": i + 1,
                    "filename": filename,
                    "author": author,
                    "title": title,
                    "chunk_id": f"{stem}_p{i + 1}_c{j}",
                })

        doc.close()

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Ошибка при обработке PDF: {e}") from e

    return all_chunks
