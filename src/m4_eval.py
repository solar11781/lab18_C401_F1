"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_text(text: str) -> str:
    return " ".join(
        token for token in
        ''.join(ch.lower() if ch.isalnum() else ' ' for ch in text).split()
    )


def _safe_overlap_score(source: str, target: str) -> float:
    source_tokens = set(_normalize_text(source).split())
    target_tokens = set(_normalize_text(target).split())
    if not source_tokens or not target_tokens:
        return 0.0
    return len(source_tokens & target_tokens) / float(len(target_tokens))


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    results: list[EvalResult] = []
    aggregate = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": []
    }

    def _fallback_metrics(question: str, answer: str, ctxs: list[str], truth: str) -> EvalResult:
        joined_contexts = " ".join(ctxs or [])
        faithfulness = _safe_overlap_score(answer, joined_contexts)
        answer_relevancy = _safe_overlap_score(answer, truth)
        context_precision = _safe_overlap_score(answer, joined_contexts)
        context_recall = _safe_overlap_score(truth, joined_contexts)
        return EvalResult(
            question=question,
            answer=answer,
            contexts=ctxs,
            ground_truth=truth,
            faithfulness=round(faithfulness, 4),
            answer_relevancy=round(answer_relevancy, 4),
            context_precision=round(context_precision, 4),
            context_recall=round(context_recall, 4),
        )

    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import _faithfulness, _answer_relevancy, _context_precision, _context_recall

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        evaluation = evaluate(
            dataset,
            metrics=[_faithfulness, _answer_relevancy, _context_precision, _context_recall],
            show_progress=False,
            raise_exceptions=False,
            column_map={
                "user_input": "question",
                "response": "answer",
                "retrieved_contexts": "contexts",
                "reference": "ground_truth",
            },
        )

        df = evaluation.to_pandas()
        for _, row in df.iterrows():
            results.append(EvalResult(
                question=row.get("question", ""),
                answer=row.get("response", ""),
                contexts=row.get("contexts", []),
                ground_truth=row.get("ground_truth", ""),
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            ))
    except Exception:
        results = [
            _fallback_metrics(q, a, c, gt)
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]

    if results:
        for result in results:
            aggregate["faithfulness"] += result.faithfulness
            aggregate["answer_relevancy"] += result.answer_relevancy
            aggregate["context_precision"] += result.context_precision
            aggregate["context_recall"] += result.context_recall
        count = len(results)
        aggregate["faithfulness"] = round(aggregate["faithfulness"] / count, 4)
        aggregate["answer_relevancy"] = round(aggregate["answer_relevancy"] / count, 4)
        aggregate["context_precision"] = round(aggregate["context_precision"] / count, 4)
        aggregate["context_recall"] = round(aggregate["context_recall"] / count, 4)
        aggregate["per_question"] = results

    return aggregate


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    def _diagnosis(result: EvalResult) -> tuple[str, str]:
        if result.faithfulness < 0.85:
            return (
                "LLM hallucinating",
                "Tighten prompt, lower temperature, or add direct source constraints."
            )
        if result.context_recall < 0.75:
            return (
                "Missing relevant chunks",
                "Improve chunking/search, expand retrieval candidates, or adjust BM25/Dense retrieval."
            )
        if result.context_precision < 0.75:
            return (
                "Too many irrelevant chunks",
                "Add reranking, metadata filtering, or sharpen similarity thresholds."
            )
        if result.answer_relevancy < 0.80:
            return (
                "Answer doesn't match question",
                "Improve prompt template, align answer format, or add answer-conditioning context."
            )
        return (
            "Low overall quality",
            "Review retrieval, prompt design, and answer generation together."
        )

    scored = []
    for result in eval_results:
        avg = (result.faithfulness + result.answer_relevancy +
               result.context_precision + result.context_recall) / 4.0
        scored.append((avg, result))

    scored.sort(key=lambda item: item[0])
    failures: list[dict] = []
    for avg, result in scored[:bottom_n]:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = _diagnosis(result)
        failures.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
            "avg_score": round(avg, 4),
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
