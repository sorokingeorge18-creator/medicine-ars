import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


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


def _ocr_image(args: tuple) -> tuple[int, str]:
    page_num, image = args
    import pytesseract
    return page_num, pytesseract.image_to_string(image, lang="rus+eng")


def process_pdf(
    file_path: str,
    text_progress_cb=None,
    ocr_progress_cb=None,
    original_filename: str = None,
) -> list[dict]:
    import pdfplumber

    # CHUNK_SIZE env var was token count; 1 token ≈ 4 chars
    max_chars = int(os.getenv("CHUNK_SIZE", 800)) * 4
    overlap_chars = int(os.getenv("CHUNK_OVERLAP", 100)) * 4

    filename = original_filename or Path(file_path).name
    stem = Path(filename).stem
    all_chunks = []

    try:
        with pdfplumber.open(file_path) as pdf:
            metadata = pdf.metadata or {}
            author = _extract_author(metadata, filename)
            title = _extract_title(metadata, filename)
            total_pages = len(pdf.pages)

            page_texts: dict[int, str] = {}
            for i, page in enumerate(pdf.pages):
                page_num = page.page_number
                if text_progress_cb:
                    text_progress_cb(i + 1, total_pages)
                try:
                    text = page.extract_text(layout=True) or ""
                except Exception:
                    text = ""
                page_texts[page_num] = text

        # OCR for blank pages (lazy-import heavy libs only when needed)
        needs_ocr = [n for n, t in page_texts.items() if len(t.strip()) < 20]

        if needs_ocr:
            try:
                import platform
                from pdf2image import convert_from_path

                tesseract_bin = shutil.which("tesseract") or "/opt/homebrew/bin/tesseract"
                pdftoppm_bin = shutil.which("pdftoppm") or "/opt/homebrew/bin/pdftoppm"
                ocr_ok = os.path.isfile(tesseract_bin) and os.path.isfile(pdftoppm_bin)

                if ocr_ok:
                    import pytesseract
                    pytesseract.pytesseract.tesseract_cmd = tesseract_bin
                    poppler_path = "/opt/homebrew/bin" if platform.system() == "Darwin" else None

                    min_p, max_p = min(needs_ocr), max(needs_ocr)
                    all_images = convert_from_path(
                        file_path, dpi=200, poppler_path=poppler_path,
                        first_page=min_p, last_page=max_p,
                    )
                    image_map = {min_p + i: img for i, img in enumerate(all_images)}
                    ocr_inputs = [(n, image_map[n]) for n in needs_ocr if n in image_map]

                    completed = 0
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        futures = {executor.submit(_ocr_image, inp): inp[0] for inp in ocr_inputs}
                        for future in as_completed(futures):
                            page_num, text = future.result()
                            page_texts[page_num] = text
                            completed += 1
                            if ocr_progress_cb:
                                ocr_progress_cb(completed, len(ocr_inputs))
            except Exception:
                pass  # OCR unavailable — continue with text-only pages

        for page_num in sorted(page_texts):
            text = page_texts[page_num]
            if not text or len(text.strip()) < 20:
                continue
            for j, chunk in enumerate(_chunk_text(text, max_chars, overlap_chars)):
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
