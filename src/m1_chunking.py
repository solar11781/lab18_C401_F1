"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, va structure-aware chunking.
So sanh voi basic chunking (baseline) de thay improvement.

Test: pytest tests/test_m1.py
"""

import glob
import io
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_CHILD_SIZE, HIERARCHICAL_PARENT_SIZE,
                    SEMANTIC_THRESHOLD)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from underthesea import sent_tokenize as vi_sent_tokenize
except ImportError:
    vi_sent_tokenize = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from PIL import Image
except ImportError:
    Image = None


SEMANTIC_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_semantic_model = None
_semantic_model_attempted = False
_ocr_reader = None
_ocr_reader_attempted = False

HEADER_RE = re.compile(r"^#{1,6}\s+\S")
LEGAL_HEADER_RE = re.compile(
    r"^(?:chuong|muc|phan|dieu|khoan)\s+(?:[ivxlcdm]+|\d+[\w./-]*)\b",
    re.IGNORECASE,
)
ROMAN_SECTION_RE = re.compile(r"^(?:[IVXLCDM]{1,7}|[A-Z])\b(?:[.)]|\s|$)")
BULLET_RE = re.compile(r"^(?:[-*+•]\s+|\d+[\.)]\s+|[a-zA-ZđĐ][\.)]\s+)")
FORM_CODE_RE = re.compile(r"^\[[0-9A-Za-z]+\]")
TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ_]+", re.UNICODE)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_pdf_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _preprocess_for_ocr(image_bytes: bytes):
    if np is None:
        return None
    if cv2 is None:
        if Image is None:
            return None
        return Image.open(io.BytesIO(image_bytes))

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.fastNlMeansDenoising(img, h=10)
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    coords = np.column_stack(np.where(img < 200))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.4:
            h, w = img.shape
            matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(
                img,
                matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
    return img


def _get_ocr_reader():
    global _ocr_reader, _ocr_reader_attempted
    if _ocr_reader is not None:
        return _ocr_reader
    if _ocr_reader_attempted or easyocr is None:
        return None

    _ocr_reader_attempted = True
    try:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "easyocr")
        os.makedirs(model_dir, exist_ok=True)
        _ocr_reader = easyocr.Reader(
            ["vi", "en"],
            gpu=False,
            verbose=False,
            model_storage_directory=model_dir,
            download_enabled=True,
        )
    except Exception:
        _ocr_reader = None
    return _ocr_reader


def _ocr_page(page) -> str:
    reader = _get_ocr_reader()
    if reader is None or fitz is None:
        return ""
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
        image_bytes = pix.tobytes("png")
        ocr_input = _preprocess_for_ocr(image_bytes)
        if ocr_input is None and Image is not None:
            ocr_input = Image.open(io.BytesIO(image_bytes))
        if ocr_input is None:
            return ""

        results = reader.readtext(
            ocr_input,
            detail=1,
            paragraph=False,
            batch_size=1,
            text_threshold=0.5,
            low_text=0.3,
        )

        lines = []
        for item in results:
            if len(item) < 3:
                continue
            _, text, confidence = item
            cleaned = _normalize_text(str(text))
            if cleaned and confidence >= 0.2:
                lines.append(cleaned)
        return "\n".join(lines).strip()
    except Exception:
        return ""


def _page_text(page) -> str:
    blocks = []
    for block in page.get_text("blocks", sort=True):
        block_text = _normalize_text(block[4])
        if block_text:
            blocks.append(block_text)
    return "\n".join(blocks).strip()


def _load_pdf(fp: str) -> str:
    if fitz is None:
        return ""
    try:
        doc = fitz.open(fp)
        total_pages = len(doc)
        pages = []
        ocr_pages = 0
        for page in doc:
            page_text = _page_text(page)
            if len(page_text) < 80:
                ocr_text = _ocr_page(page)
                if len(ocr_text) > len(page_text):
                    page_text = ocr_text
                    ocr_pages += 1
            if page_text:
                pages.append(page_text)
        doc.close()
        text = _clean_pdf_text("\n\n".join(pages))
        if text and ocr_pages:
            print(f"OCR used for {os.path.basename(fp)}: {ocr_pages}/{total_pages} pages")
        return text
    except Exception:
        return ""


def _get_semantic_model():
    global _semantic_model, _semantic_model_attempted
    if _semantic_model is not None:
        return _semantic_model
    if _semantic_model_attempted or SentenceTransformer is None:
        return None
    _semantic_model_attempted = True
    try:
        _semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME, local_files_only=True)
    except TypeError:
        _semantic_model = None
    except Exception:
        _semantic_model = None
    return _semantic_model


def _is_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if HEADER_RE.match(stripped) or LEGAL_HEADER_RE.match(stripped):
        return True
    if FORM_CODE_RE.match(stripped):
        return False
    if stripped.isupper() and 3 <= len(stripped) <= 120:
        return True
    if ROMAN_SECTION_RE.match(stripped) and len(stripped.split()) <= 16:
        return True
    return False


def _is_list_like(line: str) -> bool:
    stripped = line.strip()
    return bool(
        BULLET_RE.match(stripped)
        or FORM_CODE_RE.match(stripped)
        or re.match(r"^\d+\s+\S", stripped)
    )


def _is_table_like(line: str) -> bool:
    stripped = line.strip()
    return bool(
        FORM_CODE_RE.search(stripped)
        or re.search(r"\b\d{1,3}(?:[.,]\d{3}){1,}\b", stripped)
        or re.search(r"\s{2,}", line)
    )


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in TOKEN_RE.findall(text) if len(tok) > 1}


def _lexical_similarity(a: str, b: str) -> float:
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b) / max(1, min(len(tokens_a), len(tokens_b)))

    norm_a = re.sub(r"\s+", " ", a.lower())
    norm_b = re.sub(r"\s+", " ", b.lower())
    trigrams_a = {norm_a[i:i + 3] for i in range(max(0, len(norm_a) - 2))} or {norm_a}
    trigrams_b = {norm_b[i:i + 3] for i in range(max(0, len(norm_b) - 2))} or {norm_b}
    dice = (2 * len(trigrams_a & trigrams_b)) / max(1, len(trigrams_a) + len(trigrams_b))
    return 0.7 * overlap + 0.3 * dice


def _cosine_sim(a, b) -> float:
    if np is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _split_sentences_regex(text: str) -> list[str]:
    sentences = []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", _normalize_text(text)) if p.strip()]
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) > 1 and any(_is_list_like(line) or _is_table_like(line) for line in lines):
            sentences.extend(lines)
            continue
        parts = re.split(r"(?<=[.!?;:])\s+|\n", paragraph)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def _split_sentences(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    if vi_sent_tokenize is not None:
        try:
            sentences = []
            for paragraph in re.split(r"\n{2,}", normalized):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
                if len(lines) > 1 and any(_is_list_like(line) or _is_table_like(line) for line in lines):
                    sentences.extend(lines)
                else:
                    sentences.extend([s.strip() for s in vi_sent_tokenize(paragraph) if s.strip()])
            if sentences:
                return sentences
        except Exception:
            pass
    return _split_sentences_regex(normalized)


def _merge_orphan_headers(units: list[str]) -> list[str]:
    merged = []
    idx = 0
    while idx < len(units):
        current = units[idx].strip()
        if _is_header(current) and idx + 1 < len(units):
            merged.append(f"{current}\n{units[idx + 1].strip()}".strip())
            idx += 2
            continue
        merged.append(current)
        idx += 1
    return [item for item in merged if item]


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text
    while len(remaining) > max_chars:
        cut = remaining.rfind(" ", 0, max_chars)
        if cut < max_chars * 0.6:
            cut = max_chars
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _split_long_unit(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = _split_sentences(text)
    if len(sentences) > 1:
        parts = []
        current = []
        current_len = 0
        for sentence in sentences:
            sentence = sentence.strip()
            extra = len(sentence) + (2 if current else 0)
            if current and current_len + extra > max_chars:
                parts.append(" ".join(current).strip())
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += extra
        if current:
            parts.append(" ".join(current).strip())
        expanded = []
        for part in parts:
            if len(part) <= max_chars:
                expanded.append(part)
            else:
                expanded.extend(_hard_wrap(part, max_chars))
        return expanded

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        expanded = []
        for line in lines:
            expanded.extend(_hard_wrap(line, max_chars))
        return expanded

    return _hard_wrap(text, max_chars)


def _text_units(text: str, max_chars: int) -> list[str]:
    units = []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", _normalize_text(text)) if p.strip()]
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) > 1 and any(_is_list_like(line) or _is_table_like(line) or _is_header(line) for line in lines):
            for line in lines:
                units.extend(_split_long_unit(line, max_chars))
        else:
            units.extend(_split_long_unit(paragraph, max_chars))
    return [unit for unit in units if unit.strip()]


def _pack_units(units: list[str], max_chars: int, overlap_units: int = 0) -> list[str]:
    cleaned = []
    for unit in units:
        cleaned.extend(_split_long_unit(unit, max_chars))
    cleaned = [unit.strip() for unit in cleaned if unit.strip()]
    if not cleaned:
        return []

    chunks = []
    idx = 0
    while idx < len(cleaned):
        current = []
        current_len = 0
        start_idx = idx

        while idx < len(cleaned):
            candidate = cleaned[idx]
            extra = len(candidate) + (2 if current else 0)
            if current and current_len + extra > max_chars:
                break
            current.append(candidate)
            current_len += extra
            idx += 1

        if not current:
            current = [cleaned[idx]]
            idx += 1

        chunks.append("\n\n".join(current).strip())
        if idx >= len(cleaned):
            break
        if overlap_units > 0:
            idx = max(start_idx + 1, idx - overlap_units)

    return chunks


def _structure_sections(text: str) -> list[tuple[str, str]]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    sections = []
    current_header = ""
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if current_header or body:
            sections.append((current_header.strip(), body))

    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        if _is_header(line):
            if current_header and not any(part.strip() for part in current_lines):
                current_header = f"{current_header}\n{line}".strip()
            elif current_header or any(part.strip() for part in current_lines):
                flush()
                current_header = line
                current_lines = []
            else:
                current_header = line
        else:
            current_lines.append(line)

    flush()
    return sections


def _split_section_if_needed(header: str, body: str, max_chars: int) -> list[str]:
    full_text = f"{header}\n{body}".strip() if header else body.strip()
    if not full_text:
        return []
    if len(full_text) <= max_chars:
        return [full_text]

    body_units = _text_units(body or header, max_chars=max(120, max_chars - min(len(header), 120)))
    packed_body = _pack_units(body_units, max_chars=max(120, max_chars - min(len(header), 120)))
    chunks = []
    for part in packed_body:
        chunk_text = f"{header}\n{part}".strip() if header else part
        chunks.append(chunk_text)
    return chunks or [full_text]


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load markdown, text, and PDF files from data/."""
    docs = []
    if not os.path.isdir(data_dir):
        return docs

    for pattern in ("*.md", "*.txt"):
        for fp in sorted(glob.glob(os.path.join(data_dir, pattern))):
            try:
                with open(fp, encoding="utf-8") as f:
                    text = _normalize_text(f.read())
                if text:
                    docs.append({
                        "text": text,
                        "metadata": {"source": os.path.basename(fp), "type": "text"},
                    })
            except Exception:
                continue

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _load_pdf(fp)
        if text:
            docs.append({
                "text": text,
                "metadata": {"source": os.path.basename(fp), "type": "pdf"},
            })

    return docs


# Baseline: Basic Chunking (de so sanh)


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\n\n).
    Day la baseline - KHONG phai muc tieu cua module nay.
    (Da implement san)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# Strategy 1: Semantic Chunking


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity - nhom cau cung chu de.
    Tot hon basic vi khong cat giua y.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Duoi threshold -> tach chunk moi.
        metadata: Metadata gan vao moi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    units = _merge_orphan_headers(_split_sentences(text))
    if not units:
        return []

    model = _get_semantic_model()
    embeddings = None
    if model is not None and len(units) > 1:
        try:
            embeddings = model.encode(units, show_progress_bar=False)
        except Exception:
            embeddings = None

    effective_threshold = max(0.20, min(float(threshold), 0.95))
    min_chunk_chars = 140
    max_chunk_chars = 900

    grouped_texts = []
    current_group = [units[0]]
    current_len = len(units[0])

    for i in range(1, len(units)):
        prev_unit = units[i - 1]
        current_unit = units[i]

        if embeddings is not None and np is not None:
            sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        else:
            sim = _lexical_similarity(prev_unit, current_unit)
            effective_threshold = min(effective_threshold, 0.45)

        force_break = _is_header(current_unit)
        too_big = current_len + len(current_unit) + 1 > max_chunk_chars
        low_similarity = sim < effective_threshold and current_len >= min_chunk_chars

        if (force_break and current_group) or too_big or low_similarity:
            grouped_texts.append(" ".join(current_group).strip())
            current_group = [current_unit]
            current_len = len(current_unit)
        else:
            current_group.append(current_unit)
            current_len += len(current_unit) + 1

    if current_group:
        grouped_texts.append(" ".join(current_group).strip())

    merged_texts = []
    for group_text in grouped_texts:
        if merged_texts and len(group_text) < 80:
            merged_texts[-1] = f"{merged_texts[-1]} {group_text}".strip()
        else:
            merged_texts.append(group_text)

    chunks = []
    for idx, chunk_text in enumerate(merged_texts):
        sentence_count = len(_split_sentences(chunk_text))
        chunks.append(Chunk(
            text=chunk_text,
            metadata={
                **metadata,
                "chunk_index": idx,
                "strategy": "semantic",
                "sentence_count": sentence_count,
            },
        ))
    return chunks


# Strategy 2: Hierarchical Chunking


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) -> return parent (context).
    Day la default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gan vao moi chunk.

    Returns:
        (parents, children) - moi child co parent_id link den parent.
    """
    metadata = metadata or {}

    structure_chunks = chunk_structure_aware(text, metadata={})
    base_units = [chunk.text for chunk in structure_chunks] or _text_units(text, max_chars=parent_size)
    parent_texts = _pack_units(base_units, max_chars=parent_size, overlap_units=0)
    if not parent_texts:
        parent_texts = _hard_wrap(_normalize_text(text), parent_size)

    parents = []
    children = []

    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{parent_index}"
        parents.append(Chunk(
            text=parent_text,
            metadata={
                **metadata,
                "chunk_index": parent_index,
                "chunk_type": "parent",
                "parent_id": parent_id,
            },
            parent_id=parent_id,
        ))

        child_units = _text_units(parent_text, max_chars=child_size)
        child_texts = _pack_units(child_units, max_chars=child_size, overlap_units=1)
        if not child_texts:
            child_texts = _hard_wrap(parent_text, child_size)

        for child_index, child_text in enumerate(child_texts):
            children.append(Chunk(
                text=child_text,
                metadata={
                    **metadata,
                    "chunk_type": "child",
                    "chunk_index": child_index,
                    "parent_id": parent_id,
                },
                parent_id=parent_id,
            ))

    return parents, children


# Strategy 3: Structure-Aware Chunking


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers -> chunk theo logical structure.
    Giu nguyen tables, code blocks, lists - khong cat giua chung.

    Args:
        text: Markdown text.
        metadata: Metadata gan vao moi chunk.

    Returns:
        List of Chunk objects, moi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    sections = _structure_sections(text)
    if not sections:
        normalized = _normalize_text(text)
        return [Chunk(
            text=normalized,
            metadata={**metadata, "section": "", "strategy": "structure", "chunk_index": 0},
        )] if normalized else []

    chunks = []
    for header, body in sections:
        for chunk_text in _split_section_if_needed(header, body, max_chars=1400):
            section_name = header.strip() if header else ""
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "section": section_name,
                    "strategy": "structure",
                    "chunk_index": len(chunks),
                },
            ))

    return chunks


# A/B Test: Compare All Strategies


def _length_stats(lengths: list[int]) -> tuple[int, int, int]:
    if not lengths:
        return 0, 0, 0
    return int(sum(lengths) / len(lengths)), min(lengths), max(lengths)


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    stats = {
        "basic": {"chunks": 0, "lengths": []},
        "semantic": {"chunks": 0, "lengths": []},
        "hierarchical": {"parents": 0, "children": 0, "lengths": []},
        "structure": {"chunks": 0, "lengths": []},
    }

    for doc in documents:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})

        basic_chunks = chunk_basic(text, metadata=metadata)
        semantic_chunks = chunk_semantic(text, metadata=metadata)
        parent_chunks, child_chunks = chunk_hierarchical(text, metadata=metadata)
        structure_chunks = chunk_structure_aware(text, metadata=metadata)

        stats["basic"]["chunks"] += len(basic_chunks)
        stats["basic"]["lengths"].extend(len(chunk.text) for chunk in basic_chunks)

        stats["semantic"]["chunks"] += len(semantic_chunks)
        stats["semantic"]["lengths"].extend(len(chunk.text) for chunk in semantic_chunks)

        stats["hierarchical"]["parents"] += len(parent_chunks)
        stats["hierarchical"]["children"] += len(child_chunks)
        stats["hierarchical"]["lengths"].extend(len(chunk.text) for chunk in child_chunks)

        stats["structure"]["chunks"] += len(structure_chunks)
        stats["structure"]["lengths"].extend(len(chunk.text) for chunk in structure_chunks)

    results = {}
    for strategy, values in stats.items():
        avg_len, min_len, max_len = _length_stats(values["lengths"])
        if strategy == "hierarchical":
            chunk_repr = f"{values['parents']}p/{values['children']}c"
        else:
            chunk_repr = str(values["chunks"])
        results[strategy] = {
            "Chunks": chunk_repr,
            "Avg Len": avg_len,
            "Min": min_len,
            "Max": max_len,
        }

    header = f"{'Strategy':<15} | {'Chunks':<10} | {'Avg Len':>7} | {'Min':>5} | {'Max':>5}"
    print(header)
    print("-" * len(header))
    for strategy, values in results.items():
        print(
            f"{strategy:<15} | {values['Chunks']:<10} | "
            f"{values['Avg Len']:>7} | {values['Min']:>5} | {values['Max']:>5}"
        )

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
