# Individual Reflection — Lab 18

**Tên:** Bùi Trần Gia Bảo - 2A202600009
**Module phụ trách:** M3 - Reranking, M5 - Enrichment Pipeline and pipeline integration

---

## 1. Đóng góp kỹ thuật

- Module đã implement:
  - M3: Reranking trong `src/m3_rerank.py`
  - M5: Enrichment Pipeline trong `src/m5_enrichment.py`
  - Tích hợp production pipeline trong `src/pipeline.py`
- Các hàm/class chính đã viết:
  - Trong M3:
    - `CrossEncoderReranker._load_model()`
    - `CrossEncoderReranker.rerank()`
    - `FlashrankReranker.rerank()`
    - `benchmark_reranker()`
    - sử dụng `RerankResult` để trả về kết quả sau rerank
  - Trong M5:
    - `_clean_text()`
    - `_split_sentences()`
    - `_llm_chat()`
    - `summarize_chunk()`
    - `generate_hypothesis_questions()`
    - `contextual_prepend()`
    - `extract_metadata()`
    - `enrich_chunks()`
    - sử dụng `EnrichedChunk` để lưu original text, enriched text, summary, hypothesis questions và metadata
  - Trong pipeline:
    - hỗ trợ flow chunking → enrichment → hybrid search → rerank → LLM generate → RAGAS evaluation
- Số tests pass:
  - M3: 5/5 tests
  - M5: 10/10 tests
  - Toàn bộ pytest: 37/37 tests passed

## 2. Kiến thức học được

- Khái niệm mới nhất:
  - Tôi đã học rõ hơn về production RAG pipeline, đặc biệt là vai trò của reranking sau retrieval.
  - Với M3: hiểu cross-encoder reranker dùng query-document pair để sắp xếp lại các retrieved chunks, giúp đưa context liên quan hơn lên top.
  - Với M5: học thêm các enrichment techniques như summarization, HyQA, contextual prepend và auto metadata. Các technique này làm giàu chunk trước khi embedding để giảm vocabulary gap và cải thiện retrieval.
- Điều bất ngờ nhất:
  - Điều bất ngờ nhất là unit tests pass chưa có nghĩa là end-to-end pipeline đã ổn. Khi chạy thực tế vẫn có thể gặp issue từ Qdrant API version, OCR extraction của PDF, hoặc RAGAS evaluation warning.
  - Em cũng thấy RAGAS có thể in nhiều warning/error trong quá trình chạy nhưng cuối cùng vẫn trả về metric numbers, nên cần kiểm tra kỹ file report thay vì chỉ nhìn terminal output.
- Kết nối với bài giảng (slide nào):
  - Slide 22/42: **Reranking — Highest ROI Optimization**. Slide này liên quan tới M3 em làm: retrieve top-20 → rerank → keep top-3 → LLM generate, và dùng reranker như `bge-reranker-v2-m3`.
  - Slide 14/42: **Enrichment Techniques — 4 kỹ thuật chính**. Slide này liên quan tới M5 em làm, gồm summarize chunk, HyQA / generate câu hỏi, contextual prepend và auto metadata.
  - Slide 26/42: **RAGAS Diagnostic — Score thấp thì fix ở đâu?**. Slide này liên quan tới phần em hỗ trợ chạy evaluation/report và đọc lỗi pipeline, đặc biệt là cách nhìn context precision/recall, faithfulness và answer relevancy để debug.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất:
  - Khó khăn lớn nhất là debug các lỗi chỉ xuất hiện khi chạy thật, không xuất hiện trong unit tests.
  - Trong M3, cần làm reranker trả về đúng `RerankResult`, sort theo `rerank_score`, và vẫn có fallback khi model ngoài không load được.
  - Trong M5, cần làm các hàm enrichment hoạt động được cả khi có OpenAI API và khi không có API, để tests vẫn ổn định.
  - Khi chạy baseline/evaluation thì gặp thêm vấn đề PDF cần OCR mới extract được đủ nội dung, và RAGAS có nhiều warning từ dependency.
- Cách giải quyết:
  - Với M3: implement reranking theo hướng ưu tiên model ngoài nếu load được, và có lexical fallback deterministic để pipeline không bị crash.
  - Với M5: thêm fallback logic: summary có thể dùng extractive summary, HyQA có thể generate câu hỏi theo keyword, contextual prepend vẫn giữ original text, metadata extraction có rule-based fallback.
  - Với pipeline: ghép M5 enrichment vào trước indexing, sau đó dùng hybrid search và M3 reranker trước khi generate answer.
- Thời gian debug:
  - Khoảng vài tiếng, chủ yếu dành cho chạy tests, kiểm tra TODO markers, chạy baseline, đọc lỗi terminal và xác định lỗi nào là dependency warning, lỗi nào ảnh hưởng đến report thật.

## 4. Nếu làm lại

- Sẽ làm khác điều gì:
  - Sẽ chạy end-to-end sớm hơn, không chỉ chạy unit tests. Cụ thể là chạy `pytest`, sau đó chạy `python naive_baseline.py`, `python main.py`, và `python check_lab.py` để phát hiện lỗi report/pipeline sớm hơn.
  - Kiểm tra sớm việc PDF có extract được text hay không, vì nếu thiếu dữ liệu từ PDF thì RAGAS score và failure analysis sẽ không đáng tin cậy.
- Module nào muốn thử tiếp:
  - Tôi muốn thử tiếp M2 Hybrid Search để hiểu sâu hơn BM25 + dense retrieval + RRF ảnh hưởng thế nào đến context precision/context recall.
  - Tôi cũng muốn tiếp tục optimize M5 để so sánh rõ hơn contextual prepend và HyQA có cải thiện retrieval/RAGAS scores bao nhiêu.

## 5. Tự đánh giá

| Tiêu chí        | Tự chấm (1-5) |
| --------------- | ------------- |
| Hiểu bài giảng  | 4             |
| Code quality    | 4             |
| Teamwork        | 4             |
| Problem solving | 4             |
