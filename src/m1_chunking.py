"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os
import sys
import glob
import re
import numpy as np
from dataclasses import dataclass, field

# Thử import sentence_transformers, nếu chưa cài đặt thì gán None để không lỗi ngay khi load module
try:
    from sentence_transformers import SentenceTransformer
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    semantic_model = None

# Mock config import (hoặc giữ nguyên đường dẫn import của bạn)
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE, SEMANTIC_THRESHOLD)

# Giá trị mặc định (Fallback trong trường hợp không có file config)
DATA_DIR = "data"
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/."""
    docs = []
    # Đảm bảo thư mục tồn tại để không văng lỗi khi test
    if not os.path.exists(data_dir):
        return docs
        
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────

def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
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


# ─── Strategy 1: Semantic Chunking ───────────────────────

def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    if not semantic_model:
        raise RuntimeError("Cần cài đặt sentence-transformers: pip install sentence-transformers")

    # 1. Split text into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    # 2. Encode sentences
    embeddings = semantic_model.encode(sentences)

    # 3. Helper function for Cosine Similarity
    def cosine_sim(a, b):
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    # 4. Group sentences
    chunks = []
    current_group = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i-1], embeddings[i])
        
        # Nếu similarity dưới ngưỡng, cắt chunk mới
        if sim < threshold:
            chunk_text = " ".join(current_group)
            chunks.append(Chunk(text=chunk_text, metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}))
            current_group = []
            
        current_group.append(sentences[i])

    # 5. Add the last group
    if current_group:
        chunk_text = " ".join(current_group)
        chunks.append(Chunk(text=chunk_text, metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────

def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.
    """
    metadata = metadata or {}
    parents_list = []
    children_list = []
    
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # 1. Split text into parents
    current_parent = ""
    parent_index = 0

    for para in paragraphs:
        if len(current_parent) + len(para) > parent_size and current_parent:
            pid = f"parent_{parent_index}"
            # THÊM "parent_id": pid vào metadata
            parents_list.append(Chunk(
                text=current_parent.strip(), 
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid}, 
                parent_id=pid
            ))
            current_parent = ""
            parent_index += 1
        current_parent += para + "\n\n"

    # Don't forget last parent
    if current_parent.strip():
        pid = f"parent_{parent_index}"
        # THÊM "parent_id": pid vào metadata
        parents_list.append(Chunk(
            text=current_parent.strip(), 
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid}, 
            parent_id=pid
        ))

    # 2. Split each parent into children
    for parent in parents_list:
        p_text = parent.text
        start = 0
        
        while start < len(p_text):
            end = min(start + child_size, len(p_text))
            child_text = p_text[start:end].strip()
            
            if child_text:
                children_list.append(Chunk(
                    text=child_text, 
                    metadata={**metadata, "chunk_type": "child"}, 
                    parent_id=parent.parent_id
                ))
            start += child_size

    return parents_list, children_list


# ─── Strategy 3: Structure-Aware Chunking ────────────────

def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    
    # 1. Split by markdown headers (H1, H2, H3)
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    
    chunks = []
    current_header = ""
    current_content = ""

    # 2. Pair headers with their content
    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            if current_content.strip() or current_header.strip():
                full_text = f"{current_header}\n{current_content}".strip()
                if full_text:
                    chunks.append(Chunk(
                        text=full_text,
                        metadata={**metadata, "section": current_header.strip(), "strategy": "structure"}
                    ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    # Xử lý đoạn section cuối cùng
    if current_content.strip() or current_header.strip():
        full_text = f"{current_header}\n{current_content}".strip()
        if full_text:
            chunks.append(Chunk(
                text=full_text,
                metadata={**metadata, "section": current_header.strip(), "strategy": "structure"}
            ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────

def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    """
    stats = {
        "basic": {"chunks": 0, "lengths": []},
        "semantic": {"chunks": 0, "lengths": []},
        "hierarchical": {"parents": 0, "children": 0, "lengths": []},
        "structure": {"chunks": 0, "lengths": []}
    }

    # 1. Process all documents
    for doc in documents:
        text = doc["text"]
        meta = doc["metadata"]

        # Basic
        b_chunks = chunk_basic(text, chunk_size=500, metadata=meta)
        stats["basic"]["chunks"] += len(b_chunks)
        stats["basic"]["lengths"].extend([len(c.text) for c in b_chunks])

        # Semantic
        if semantic_model:
            s_chunks = chunk_semantic(text, threshold=SEMANTIC_THRESHOLD, metadata=meta)
            stats["semantic"]["chunks"] += len(s_chunks)
            stats["semantic"]["lengths"].extend([len(c.text) for c in s_chunks])

        # Hierarchical
        parents, children = chunk_hierarchical(text, metadata=meta)
        stats["hierarchical"]["parents"] += len(parents)
        stats["hierarchical"]["children"] += len(children)
        stats["hierarchical"]["lengths"].extend([len(c.text) for c in parents])

        # Structure
        st_chunks = chunk_structure_aware(text, metadata=meta)
        stats["structure"]["chunks"] += len(st_chunks)
        stats["structure"]["lengths"].extend([len(c.text) for c in st_chunks])

    # 2. Compile Results
    results = {}
    for strategy, data in stats.items():
        lengths = data["lengths"] or [0]
        avg_len = sum(lengths) // len(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        
        if strategy == "hierarchical":
            chunks_str = f"{data['parents']}p/{data['children']}c"
        else:
            chunks_str = str(data["chunks"])
            
        results[strategy] = {
            "Chunks": chunks_str,
            "Avg Len": avg_len,
            "Min": min_len,
            "Max": max_len
        }

    # 3. Print comparison table
    print(f"\n{'Strategy':<15} | {'Chunks':<10} | {'Avg Len':<7} | {'Min':<5} | {'Max':<5}")
    print("-" * 55)
    for strategy, res in results.items():
        print(f"{strategy:<15} | {res['Chunks']:<10} | {res['Avg Len']:<7} | {res['Min']:<5} | {res['Max']:<5}")
    print("\n")

    return results


if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        print("Không tìm thấy file markdown nào trong thư mục. Tạo một file dummy để test...")
        docs = [{"text": "# Header 1\n\nThis is paragraph 1.\n\nThis is paragraph 2.\n\n## Header 2\n\nContent for header 2.", "metadata": {"source": "dummy.md"}}]
        
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)