"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _split_sentences(text: str) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def _llm_chat(system_prompt: str, user_prompt: str, max_tokens: int = 150) -> str:
    """Use OpenAI when available; otherwise return empty string for fallback."""
    if not OPENAI_API_KEY:
        return ""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""

@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return ""

    llm_summary = _llm_chat(
        "Tóm tắt đoạn văn sau trong tối đa 2 câu ngắn gọn bằng tiếng Việt.",
        cleaned,
        max_tokens=120,
    )
    if llm_summary:
        return llm_summary

    sentences = _split_sentences(cleaned)
    if not sentences:
        return cleaned[:300]
    return " ".join(sentences[:2])


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    # Implement hypothesis question generation
    # 1. from openai import OpenAI
    #    client = OpenAI()
    # 2. resp = client.chat.completions.create(
    #        model="gpt-4o-mini",
    #        messages=[
    #            {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
    #            {"role": "user", "content": text},
    #        ],
    #        max_tokens=200,
    #    )
    # 3. questions = resp.choices[0].message.content.strip().split("\n")
    # 4. return [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()]
    #
    # Tại sao: User hỏi "nghỉ phép bao nhiêu ngày?" nhưng doc viết
    # "12 ngày làm việc mỗi năm" → vocabulary gap. HyQA bridge gap này
    # bằng cách index câu hỏi "Nhân viên được nghỉ bao nhiêu ngày?" cùng chunk.
    cleaned = _clean_text(text)
    if not cleaned or n_questions <= 0:
        return []

    llm_output = _llm_chat(
        f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
        "Trả về mỗi câu hỏi trên một dòng, không đánh số.",
        cleaned,
        max_tokens=180,
    )
    if llm_output:
        questions = []
        for line in llm_output.splitlines():
            question = re.sub(r"^[\s\-•*\d.)]+", "", line).strip()
            if question:
                questions.append(question if question.endswith("?") else f"{question}?")
        return questions[:n_questions]

    words = re.findall(r"[0-9A-Za-zÀ-ỹĐđ_/.-]+", cleaned)
    keywords: list[str] = []
    for word in words:
        normalized = word.strip(".,;:()[]{}").lower()
        if len(normalized) < 3 or normalized in {
            "của", "cho", "the", "and", "hoặc", "trong", "được", "theo", "này",
        }:
            continue
        if normalized not in keywords:
            keywords.append(normalized)

    questions = []
    if keywords:
        questions.append(f"Đoạn tài liệu này cung cấp thông tin gì về {keywords[0]}?")
    if len(keywords) > 1:
        questions.append(f"Những quy định hoặc số liệu nào liên quan đến {keywords[1]}?")
    if re.search(r"\d", cleaned):
        questions.append("Các giá trị, thời hạn hoặc số liệu quan trọng trong đoạn này là gì?")
    if not questions:
        questions.append("Đoạn tài liệu này trả lời câu hỏi gì?")

    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    cleaned = _clean_text(text)
    title = _clean_text(document_title)

    if not cleaned:
        return ""

    prompt = f"Tài liệu: {title or 'không rõ'}\n\nĐoạn văn:\n{cleaned}"
    context = _llm_chat(
        "Viết đúng 1 câu ngắn mô tả đoạn văn này nằm trong tài liệu nào "
        "và nói về chủ đề gì. Chỉ trả về 1 câu.",
        prompt,
        max_tokens=80,
    )

    if not context:
        context = f"Trích từ {title}." if title else "Đoạn trích từ tài liệu nguồn."

    return f"{context}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return {"topic": "", "entities": [], "category": "general", "language": "vi"}

    llm_output = _llm_chat(
        'Trích xuất metadata từ đoạn văn. Chỉ trả về JSON hợp lệ với schema: '
        '{"topic": "...", "entities": ["..."], '
        '"category": "policy|hr|it|finance|general", "language": "vi|en"}.',
        cleaned,
        max_tokens=150,
    )

    if llm_output:
        try:
            parsed = json.loads(llm_output.strip().strip("`"))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    lower = cleaned.lower()

    if any(term in lower for term in ["mật khẩu", "vpn", "aes", "wireguard", "it"]):
        category = "it"
    elif any(term in lower for term in ["lương", "thưởng", "chi phí", "hoàn tiền", "finance"]):
        category = "finance"
    elif any(term in lower for term in ["nhân viên", "nghỉ phép", "thử việc", "hr"]):
        category = "hr"
    elif any(term in lower for term in ["chính sách", "quy định", "policy"]):
        category = "policy"
    else:
        category = "general"

    topic_terms = [
        "nghỉ phép",
        "mật khẩu",
        "vpn",
        "thử việc",
        "bảo mật",
        "lương",
        "nhân viên",
        "chính sách",
    ]
    topic = next((term for term in topic_terms if term in lower), "general")

    language = (
        "vi"
        if re.search(r"[ăâđêôơưáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ]", lower)
        else "en"
    )

    entities = sorted(
        set(
            re.findall(
                r"\b[A-ZÀ-ỸĐ][\wÀ-ỹĐđ-]*(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹĐđ-]*)*",
                cleaned,
            )
        )
    )
    numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", cleaned)

    return {
        "topic": topic,
        "entities": (entities + numbers)[:10],
        "category": category,
        "language": language,
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    method_set = set(methods)
    use_full = "full" in method_set
    enriched: list[EnrichedChunk] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        metadata = dict(chunk.get("metadata", {}) or {})

        if not text:
            continue

        summary = summarize_chunk(text) if use_full or "summary" in method_set else ""
        questions = (
            generate_hypothesis_questions(text)
            if use_full or "hyqa" in method_set
            else []
        )
        auto_meta = extract_metadata(text) if use_full or "metadata" in method_set else {}

        if use_full or "contextual" in method_set:
            enriched_text = contextual_prepend(text, metadata.get("source", ""))
        else:
            enriched_text = text

        additions: list[str] = []
        if summary:
            additions.append(f"Tóm tắt: {summary}")
        if questions:
            additions.append("Câu hỏi có thể trả lời: " + " | ".join(questions))

        if additions:
            enriched_text = f"{enriched_text}\n\n" + "\n".join(additions)

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**metadata, **auto_meta},
                method="+".join(methods),
            )
        )

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
