import os
import re
from pathlib import Path


def _extract_author(metadata, filename: str) -> str:
    for val in (getattr(metadata, "author", None), getattr(metadata, "creator", None)):
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    stem = Path(filename).stem
    parts = re.split(r"[_\-\s]+", stem)
    for part in parts:
        if len(part) >= 3 and part[0].isupper() and part.isalpha():
            return part
    return "Автор неизвестен"


def _extract_title(metadata, filename: str) -> str:
    val = getattr(metadata, "title", None)
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
    from pypdf import PdfReader

    max_chars = int(os.getenv("CHUNK_SIZE", 800)) * 4
    overlap_chars = int(os.getenv("CHUNK_OVERLAP", 100)) * 4

    filename = original_filename or Path(file_path).name
    stem = Path(filename).stem
    all_chunks = []

    try:
        reader = PdfReader(file_path)
        metadata = reader.metadata or {}
        author = _extract_author(metadata, filename)
        title = _extract_title(metadata, filename)
        total_pages = len(reader.pages)

        for i, page in enumerate(reader.pages):
            if text_progress_cb:
                text_progress_cb(i + 1, total_pages)
            try:
                text = page.extract_text() or ""
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

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Ошибка при обработке PDF: {e}") from e

    return all_chunks
