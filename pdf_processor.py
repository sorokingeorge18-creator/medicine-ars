import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pdfplumber
import tiktoken

try:
    import pytesseract
    from pdf2image import convert_from_path

    _tesseract = shutil.which("tesseract") or "/opt/homebrew/bin/tesseract"
    _pdftoppm = shutil.which("pdftoppm") or "/opt/homebrew/bin/pdftoppm"

    OCR_AVAILABLE = os.path.isfile(_tesseract) and os.path.isfile(_pdftoppm)

    if OCR_AVAILABLE:
        pytesseract.pytesseract.tesseract_cmd = _tesseract
        import platform
        _POPPLER_PATH = "/opt/homebrew/bin" if platform.system() == "Darwin" else None
except ImportError:
    OCR_AVAILABLE = False
    _POPPLER_PATH = None


enc = tiktoken.get_encoding("cl100k_base")


def _extract_author(metadata: dict, filename: str) -> str:
    for key in ("Author", "Creator", "author", "creator"):
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
    title = metadata.get("Title") or metadata.get("title")
    if title and isinstance(title, str) and title.strip():
        return title.strip()
    return Path(filename).stem.replace("_", " ")


def _chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> list[str]:
    tokens = enc.encode(text)
    if not tokens:
        return []
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += max_tokens - overlap
    return chunks


def _ocr_image(args: tuple) -> tuple[int, str]:
    """OCR one page image. Returns (page_num, text)."""
    page_num, image = args
    return page_num, pytesseract.image_to_string(image, lang="rus+eng")


def process_pdf(
    file_path: str,
    text_progress_cb=None,
    ocr_progress_cb=None,
    original_filename: str = None,
) -> list[dict]:
    """
    Parse a PDF and return text chunks with metadata.

    text_progress_cb(current, total) — called during pdfplumber phase.
    ocr_progress_cb(current, total)  — called as OCR pages complete.
    original_filename — real name when file_path is a temp file.
    """
    max_tokens = int(os.getenv("CHUNK_SIZE", 800))
    overlap = int(os.getenv("CHUNK_OVERLAP", 100))

    filename = original_filename or Path(file_path).name
    stem = Path(filename).stem
    all_chunks = []

    try:
        with pdfplumber.open(file_path) as pdf:
            metadata = pdf.metadata or {}
            author = _extract_author(metadata, filename)
            title = _extract_title(metadata, filename)
            total_pages = len(pdf.pages)

            # Phase 1: extract text with pdfplumber
            page_texts: dict[int, str] = {}
            for i, page in enumerate(pdf.pages):
                page_num = page.page_number
                if text_progress_cb:
                    text_progress_cb(i + 1, total_pages)
                text = ""
                try:
                    text = page.extract_text(layout=True) or ""
                except Exception:
                    pass
                page_texts[page_num] = text

        # Phase 2: parallel OCR for pages without text
        needs_ocr = [n for n, t in page_texts.items() if len(t.strip()) < 20]

        if needs_ocr and OCR_AVAILABLE:
            min_p, max_p = min(needs_ocr), max(needs_ocr)

            # Convert all needed pages in one poppler call
            all_images = convert_from_path(
                file_path,
                dpi=200,
                poppler_path=_POPPLER_PATH,
                first_page=min_p,
                last_page=max_p,
            )
            image_map = {min_p + i: img for i, img in enumerate(all_images)}
            ocr_inputs = [(n, image_map[n]) for n in needs_ocr if n in image_map]

            completed = 0
            total_ocr = len(ocr_inputs)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_ocr_image, inp): inp[0] for inp in ocr_inputs}
                for future in as_completed(futures):
                    page_num, text = future.result()
                    page_texts[page_num] = text
                    completed += 1
                    if ocr_progress_cb:
                        ocr_progress_cb(completed, total_ocr)

        # Phase 3: chunk all pages
        for page_num in sorted(page_texts):
            text = page_texts[page_num]
            if not text or len(text.strip()) < 20:
                continue
            for j, chunk in enumerate(_chunk_text(text, max_tokens, overlap)):
                all_chunks.append({
                    "text": chunk,
                    "page": page_num,
                    "filename": filename,
                    "author": author,
                    "title": title,
                    "chunk_id": f"{stem}_p{page_num}_c{j}",
                })

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Ошибка при обработке PDF: {e}") from e

    return all_chunks
