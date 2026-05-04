"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            # TODO: Load cross-encoder model
            # Option A: from FlagEmbedding import FlagReranker
            #           self._model = FlagReranker(self.model_name, use_fp16=True)
            # Option B: from sentence_transformers import CrossEncoder
            #           self._model = CrossEncoder(self.model_name)
            #
            # Fallback: try to load one of the recommended rerankers, but
            # if neither is available keep self._model as None and use the
            # lightweight lexical fallback implemented in `rerank()`.
            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(self.model_name, use_fp16=True)
            except Exception:
                try:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder(self.model_name)
                except Exception:
                    # Specialized reranker not available in the environment;
                    # leave _model as None to trigger the fallback.
                    self._model = None
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k.

        The implementation prefers a loaded cross-encoder/FlagReranker when
        available (both are attempted). If no external model can be loaded
        this function falls back to a lightweight lexical-overlap scorer.
        """
        # TODO: Implement reranking
        # 1. model = self._load_model()
        # 2. pairs = [(query, doc["text"]) for doc in documents]
        # 3. scores = model.compute_score(pairs)  # FlagReranker
        #    OR scores = model.predict(pairs)      # CrossEncoder
        # 4. Combine: [(score, doc) for score, doc in zip(scores, documents)]
        # 5. Sort by score descending
        # 6. Return top_k RerankResult(text=..., original_score=doc["score"],
        #                              rerank_score=score, metadata=doc["metadata"], rank=i)

        model = self._load_model()
        pairs = [(query, d.get("text", "")) for d in documents]

        scores = None
        # Try using a loaded model if present
        if model is not None:
            try:
                # FlagReranker provides compute_score(pairs)
                if hasattr(model, "compute_score"):
                    scores = model.compute_score(pairs)
                # CrossEncoder provides predict(pairs)
                elif hasattr(model, "predict"):
                    scores = model.predict(pairs)
                # Some wrappers may be callable
                elif callable(model):
                    scores = model(pairs)
            except Exception:
                # If model call fails for any reason, fall back to lexical scorer
                scores = None

        # Lexical fallback: simple, deterministic scorer based on token overlap
        if scores is None:
            def _tokens(s: str):
                # Extract word-like tokens preserving unicode (Vietnamese)
                return re.findall(r"\w+", s.lower(), flags=re.UNICODE)

            q_tokens = _tokens(query)
            q_token_set = set(q_tokens)
            scores = []
            for d in documents:
                d_text = d.get("text", "")
                d_tokens = _tokens(d_text)
                d_token_set = set(d_tokens)
                # Overlap ratio (how many distinct query tokens appear in doc)
                overlap = 0.0
                if q_token_set:
                    overlap = sum(1 for t in q_token_set if t in d_token_set) / len(q_token_set)
                # Numeric-match boost (e.g., '12' in both query and doc)
                num_query = re.findall(r"\d+", query)
                num_doc = re.findall(r"\d+", d_text)
                num_boost = 0.0
                if num_query and num_doc and set(num_query) & set(num_doc):
                    num_boost = 0.2
                orig = float(d.get("score", 0.0))
                # Combine signals: lexical overlap gets most weight, original
                # retrieval score acts as a tie-breaker.
                rerank_score = overlap * 0.8 + orig * 0.2 + num_boost
                scores.append(rerank_score)

        # Pair scores with documents and sort descending
        paired = list(zip(scores, documents))
        paired.sort(key=lambda x: float(x[0]) if x[0] is not None else 0.0, reverse=True)
        top = paired[: max(1, min(len(paired), top_k))]
        results = []
        for i, (score, doc) in enumerate(top, start=1):
            results.append(RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            ))
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        # TODO (optional): from flashrank import Ranker, RerankRequest
        # model = Ranker(); passages = [{"text": d["text"]} for d in documents]
        # results = model.rerank(RerankRequest(query=query, passages=passages))
        # Provide a simple deterministic lightweight re-ranker using lexical overlap
        def _tokens(s: str):
            return re.findall(r"\w+", s.lower(), flags=re.UNICODE)

        q_tokens = set(_tokens(query))
        scored = []
        for d in documents:
            d_tokens = set(_tokens(d.get("text", "")))
            overlap = 0.0
            if q_tokens:
                overlap = sum(1 for t in q_tokens if t in d_tokens) / len(q_tokens)
            orig = float(d.get("score", 0.0))
            score = overlap * 0.9 + orig * 0.1
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: max(1, min(len(scored), top_k))]
        results = []
        for i, (score, doc) in enumerate(top, start=1):
            results.append(RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            ))
        return results


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    # TODO: Implement benchmark
    # 1. times = []
    # 2. for _ in range(n_runs):
    #      start = time.perf_counter()
    #      reranker.rerank(query, documents)
    #      times.append((time.perf_counter() - start) * 1000)  # ms
    # 3. return {"avg_ms": mean(times), "min_ms": min(times), "max_ms": max(times)}
    times = []
    for _ in range(max(1, int(n_runs))):
        start = time.perf_counter()
        try:
            reranker.rerank(query, documents)
        except Exception:
            # Don't let a single run crash the benchmark
            pass
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)
    avg = sum(times) / len(times) if times else 0.0
    minimum = min(times) if times else 0.0
    maximum = max(times) if times else 0.0
    return {"avg_ms": avg, "min_ms": minimum, "max_ms": maximum}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
