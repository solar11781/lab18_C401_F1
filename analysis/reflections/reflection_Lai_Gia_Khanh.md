# Individual Reflection — Lab 18

**Tên:** Lại Gia Khánh <br>
**Module phụ trách:** M2

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M2 - Hybrid Search (BM25 + Dense + RRF)
- Các hàm/class chính đã viết:
  - `segment_vietnamese()`: Vietnamese word segmentation using underthesea
  - `BM25Search.index()`: Build BM25 index from chunks
  - `BM25Search.search()`: BM25-based search with Vietnamese tokenization
  - `DenseSearch.index()`: Index chunks into Qdrant vector database
  - `DenseSearch.search()`: Dense vector search using sentence transformers
  - `reciprocal_rank_fusion()`: Merge BM25 and dense results using RRF algorithm
- Số tests pass: 5/5 (tất cả tests của module M2)

## 2. Kiến thức học được

- Khái niệm mới nhất:
    - Hiểu rõ sự khác biệt giữa BM25 (exact match) và Dense Vector (semantic search)
    - Cách kết hợp 2 phương pháp bằng Reciprocal Rank Fusion (RRF) để tận dụng ưu điểm của cả hai
- Điều bất ngờ nhất: 
    - BM25 nếu không segment tiếng Việt thì gần như không hiệu quả
    - Dense search mạnh về semantic nhưng lại yếu với exact keyword
    - RRF rất đơn giản (không cần training) nhưng vẫn cho kết quả tốt trong production
- Kết nối với bài giảng:
    - Slide “Hybrid Search — BM25 + Dense Vector Fusion”: Hiểu pipeline kết hợp 2 nhánh search
    - Slide “BM25 vs Dense — Khi nào dùng cái nào?”: So sánh ưu/nhược điểm
    - Slide “Beyond RRF — Tensor Fusion…”: Biết thêm các hướng nâng cao (ColBERT, SPLADE, Tensor Fusion)

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: xử lý phân tách từ tiếng Việt (tokenization) và phụ thuộc vào thư viện `underthesea` trên Windows.
- Cách giải quyết: dùng `underthesea.word_tokenize(..., format="text")` cho BM25 và tách rõ phần dense để test unit không phụ thuộc Qdrant.
- Thời gian debug: ~1 giờ (cải thiện tokenization, sửa thứ tự kết quả, viết RRF).

## 4. Nếu làm lại

- Sẽ làm khác điều gì: tách tests riêng cho `DenseSearch` (mock Qdrant), thêm logging, và xử lý ngoại lệ khi Qdrant/encoder không có sẵn.
- Module nào muốn thử tiếp:
    - M3 - Reranking (Cross-encoder) để improve precision top-K
    - M4 - Evaluation (RAGAS) để hiểu rõ performance pipeline

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
