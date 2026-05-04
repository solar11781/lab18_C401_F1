# Group Report — Lab 18: Production RAG

**Nhóm:** [Tên]  
**Ngày:**

## Thành viên & Phân công

### Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Lê Duy Anh | M1: Chunking | ☑ | 8/8 |
| Lại Gia Khánh | M2: Hybrid Search | ☑ | 5/5 |
| Bùi Trần Gia Bảo | M3: Reranking | ☑ | 5/5 |
| Trương Minh Sơn | M4: Evaluation | ☑ | 4/4 |

## Kết quả RAGAS

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8898 | 0.3667 | -0.5231 |
| Answer Relevancy | 0.0511 | 0.2637 | +0.2126 |
| Context Precision | 0.4875 | 0.6417 | +0.1542 |
| Context Recall | 0.55 | 0.55 | 0.00 |

## Key Findings

1. **Biggest improvement:** **Answer Relevancy** (+0.2126) and **Context Precision** (+0.1542). The introduction of Hybrid Search and Reranking significantly improved the system's ability to retrieve the most relevant chunks for the user query.
2. **Biggest challenge:** **Faithfulness** dropped significantly (-0.5231). The Production system frequently returns "Không tìm thấy" (Not found) even when relevant information is present in the retrieved context, indicating a generation or prompt constraint issue.
3. **Surprise finding:** The **Naive Baseline** had a very high Faithfulness score (0.8898) but an extremely low Answer Relevancy (0.0511), suggesting it was very "safe" but rarely actually answered the question asked.

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production):
2. Biggest win — module nào, tại sao:
3. Case study — 1 failure, Error Tree walkthrough:
4. Next optimization nếu có thêm 1 giờ:
