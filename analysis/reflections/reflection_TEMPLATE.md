# Individual Reflection — Lab 18

**Tên:** [Your Name]  
**Module phụ trách:** M4

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M4 — RAGAS Evaluation
- Các hàm/class chính đã viết:
  - `evaluate_ragas()` in `src/m4_eval.py` to compute RAGAS metrics from question-answer-context-ground truth data
  - `failure_analysis()` in `src/m4_eval.py` to identify the worst-performing questions and map them to diagnoses and suggested fixes
  - Fixed support for fallback evaluation when `OPENAI_API_KEY` is missing, ensuring unit tests still pass
- Số tests pass: 4/4 on `pytest tests/test_m4.py`

## 2. Kiến thức học được

- Khái niệm mới nhất: RAGAS evaluation workflow and how metrics like faithfulness, answer relevancy, context precision, and context recall are computed with `ragas.evaluate()`
- Điều bất ngờ nhất: `ragas` can require `column_map` to align dataset fields with expected metric names, and live evaluation fails cleanly if API keys are absent
- Kết nối với bài giảng (slide nào): liên quan đến phần đánh giá chất lượng RAG và phân tích lỗi tương tự slide về "evaluation metrics" và "failure analysis"

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: tích hợp `ragas.evaluate()` với dữ liệu thực tế và xử lý tình huống khi môi trường không có `OPENAI_API_KEY`
- Cách giải quyết: kiểm tra trực tiếp API bằng Python, dùng fallback token-overlap để giữ module hoạt động, và sửa lỗi dữ liệu đầu vào của `test_set.json`
- Thời gian debug: khoảng 2-3 giờ tập trung để viết, kiểm tra API, và đảm bảo tests chạy sạch

## 4. Nếu làm lại

- Sẽ làm khác điều gì: tôi sẽ nâng cấp evaluation bằng cách bổ sung thêm minh chứng đánh giá ở mức câu hỏi và xây dựng trực quan report tốt hơn cho nhóm
- Module nào muốn thử tiếp: M5 để tích hợp enrichment và cải thiện chất lượng context trước khi đánh giá

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 5 |
