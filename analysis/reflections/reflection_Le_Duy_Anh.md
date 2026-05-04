# Individual Reflection - Lab 18

**Tên:** Lê Duy Anh  
**Module phụ trách:** M1

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M1 - Advanced Chunking Strategies
- Các hàm/class chính đã viết:
  - `chunk_semantic()` trong [src/m1_chunking.py] để nhóm câu theo độ tương đồng ngữ nghĩa, có fallback lexical similarity khi model embedding không sẵn sàng
  - `chunk_hierarchical()` trong [src/m1_chunking.py] để tạo parent/child chunks, gán `parent_id` và dùng overlap nhỏ cho child retrieval
  - `chunk_structure_aware()` trong [src/m1_chunking.py] để nhận diện header markdown, section pháp lý, list, table-like lines và chia chunk theo cấu trúc logic
  - `compare_strategies()` để chạy A/B giữa basic, semantic, hierarchical, structure-aware và thống kê số chunk/độ dài trung bình
  - `load_documents()` để đọc `.md`, `.txt`, `.pdf`, có thêm fallback OCR cho PDF scan khi text layer quá ít
- Số tests pass: 13/13 trên `pytest tests/test_m1.py`

## 2. Kiến thức học được

- Khái niệm mới nhất: chunking không chỉ là cắt theo độ dài, mà cần giữ được ngữ nghĩa và cấu trúc tài liệu để tăng `context_recall` và `context_precision` cho RAG
- Điều bất ngờ nhất: cùng một tài liệu nhưng chiến lược chunk khác nhau sẽ ảnh hưởng rất lớn đến chất lượng retrieve; văn bản pháp lý hợp với structure-aware, còn production RAG thường hợp với hierarchical parent-child hơn
- Kết nối với bài giảng: phần này liên hệ trực tiếp đến phần preprocessing/indexing trong pipeline RAG, đặc biệt là ý retrieve child -> return parent và trade-off giữa precision và recall

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: dữ liệu trong `data/` không đồng nhất; file Nghị định có cấu trúc pháp lý rõ ràng, còn `BCTC.pdf` gần như là scan/biểu mẫu bảng ảnh nên text layer rất nghèo
- Cách giải quyết:
  - Viết chunking theo hướng structure-first thay vì cắt cứng theo ký tự
  - Thêm heuristic nhận diện `Chương`, `Điều`, `Khoản`, header viết hoa, danh sách và dòng giống bảng để tránh cắt vô nghĩa
  - Dùng lazy loading cho sentence-transformers và fallback lexical similarity để code vẫn chạy được trong môi trường offline/lab
  - Nối thêm đường OCR fallback cho PDF scan để `load_documents()` có khả năng xử lý dữ liệu thực tế, dù package OCR trong môi trường vẫn cần cài thêm
- Thời gian debug: khoảng 3 giờ để đọc test, quan sát raw data, chỉnh logic chunking, và đảm bảo không làm vỡ `chunk_basic()`

## 4. Nếu làm lại

- Sẽ làm khác điều gì: tôi sẽ tách riêng pipeline OCR/extraction thành một lớp preprocessing độc lập, có cache text đã OCR theo từng file để tránh phải quét lại mỗi lần chạy
- Module nào muốn thử tiếp: M2 hoặc M5, vì sau khi chunking ổn định thì search và enrichment là 2 bước tiếp theo ảnh hưởng trực tiếp đến kết quả cuối cùng

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 5 |
