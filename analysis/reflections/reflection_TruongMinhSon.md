# Individual Reflection — Lab 18

**Tên:** [Trương Minh Sơn]
**Module phụ trách:** M4 — Evaluation

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** M4 — RAGAS Evaluation.
- **Các hàm/class chính đã viết:**
  - `evaluate_ragas()`: Thực hiện tính toán 4 chỉ số (Faithfulness, Relevancy, Precision, Recall) bằng thư viện Ragas.
  - `failure_analysis()`: Tự động trích xuất các câu hỏi có điểm thấp nhất để phục vụ phân tích lỗi.
  - Xây dựng file `group_report.md` và `failure_analysis.md` dựa trên kết quả từ hai báo cáo `naive_baseline_report.json` và `ragas_report.json`.
  - **Xử lý lỗi:** Viết cơ chế fallback để hệ thống không bị crash khi thiếu `OPENAI_API_KEY`, đảm bảo module vẫn pass unit test trong môi trường CI/CD.
- **Số tests pass:** 4/4 trên `pytest tests/test_m4.py`.

## 2. Kiến thức học được

- **Quy trình đánh giá RAG:** Hiểu sâu về "RAG Triad" và cách RAGAS sử dụng LLM-as-a-judge để đánh giá mà không cần quá nhiều nhãn thủ công.
- **Chỉ số RAGAS:** Cách phân biệt giữa **Context Precision** (độ chính xác của truy xuất) và **Faithfulness** (độ trung thực của câu trả lời). Trong bài này, tôi học được rằng Precision cao không đồng nghĩa với Faithfulness cao.
- **Xử lý dữ liệu:** Cách mapping các cột dữ liệu (question, contexts, answer, ground_truth) để tương thích với yêu cầu của Ragas Dataset.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn 1:** Điểm *Faithfulness* của bản Production giảm sâu (0.36). Ban đầu tôi nghi ngờ code M4 lỗi, nhưng sau khi kiểm tra log, tôi nhận ra vấn đề nằm ở việc LLM bị "ảo tưởng" hoặc trả về "Không tìm thấy" quá nhiều.
- **Khó khăn 2:** Giới hạn API Key và chi phí khi chạy đánh giá trên tập dữ liệu lớn.
- **Cách giải quyết:** Sử dụng một tập mẫu (sample) để test code trước, sau đó mới chạy toàn bộ. Bổ sung logic kiểm tra định dạng dữ liệu đầu vào (JSON) cực kỳ chặt chẽ trước khi đưa vào hàm evaluate.

## 4. Nếu làm lại (Phản hồi & Cải tiến)

Nếu có cơ hội làm lại hoặc tối ưu thêm, tôi sẽ:
- **Phân tích sâu hơn về prompt:** Thay vì chỉ báo cáo điểm số, tôi sẽ thử thay đổi Prompt của LLM Judge trong Ragas để xem kết quả có khách quan hơn đối với tiếng Việt hay không.
- **Kết hợp đánh giá thủ công (Human-in-the-loop):** Chọn ra 10% số câu hỏi để tự đánh giá và so sánh với điểm của Ragas nhằm tính toán độ tương quan (correlation).
- **Tối ưu hóa Faithfulness:** Phối hợp chặt chẽ hơn với thành viên module M1 để điều chỉnh chunking, vì tôi nhận thấy việc mất Faithfulness phần lớn do context bị cắt vụn làm LLM mất dấu thực thể (như tên người ký tờ khai).

## 5. Tóm tắt kết quả đạt được (Key Results)

Dưới đây là bảng so sánh mà tôi đã tổng hợp cho nhóm:

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.8898 | 0.3667 | -0.5231 |
| Answer Relevancy | 0.0511 | 0.2637 | +0.2126 |
| Context Precision | 0.4875 | 0.6417 | +0.1542 |

*Hệ thống Production đã cải thiện khả năng tìm kiếm thông tin nhưng cần tập trung cải thiện khả năng trích xuất chính xác.*

## 6. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|----------|---------------|---------|
| Hiểu bài giảng | 5 | Nắm vững cách vận hành của RAGAS |
| Code quality | 5 | Code sạch, có xử lý ngoại lệ và pass toàn bộ test |
| Teamwork | 4 | Hỗ trợ nhóm tổng hợp báo cáo cuối cùng |
| Problem solving | 5 | Tìm ra nguyên nhân điểm Faithfulness thấp |
