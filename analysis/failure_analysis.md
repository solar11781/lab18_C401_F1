# Failure Analysis — Lab 18: Production RAG

**Nhóm:** C401_F1
**Thành viên:** Member 1 [Lê Duy Anh] · Member 2 [Lại Gia Khánh] · Member 3 [Bùi Trần Gia Bảo] · Member 4 [Trương Minh Sơn]

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8898 | 0.3667 | -0.5231 |
| Answer Relevancy | 0.0511 | 0.2637 | +0.2126 |
| Context Precision | 0.4875 | 0.6417 | +0.1542 |
| Context Recall | 0.5500 | 0.5500 | 0.0000 |

## Bottom-5 Failures

### #1
- **Question:** Tờ khai được lập ngày nào và người ký đại diện là ai?
- **Expected:** Tờ khai được lập ngày 24 tháng 01 năm 2025; người ký/đại diện là TRỊNH THỊ SANG.
- **Got:** Không tìm thấy.
- **Worst metric:** faithfulness
- **Error Tree:** Output sai → Context đúng (thông tin có trong BCTC.pdf) → Query OK
- **Root cause:** LLM hallucinating. The model failed to extract specific fields despite relevant context being retrieved.
- **Suggested fix:** Tighten prompt, lower temperature, or add direct source constraints.

### #2
- **Question:** Tờ khai có ghi số chứng chỉ hành nghề của nhân viên đại lý thuế không?
- **Expected:** Không. Mục 'Chứng chỉ hành nghề số' trong phần Nhân viên đại lý thuế để trống.
- **Got:** Không tìm thấy.
- **Worst metric:** faithfulness
- **Error Tree:** Output sai → Context đúng? (Dữ liệu trống) → Query OK
- **Root cause:** LLM hallucinating. The model struggles with confirming the absence of information in a form.
- **Suggested fix:** Tighten prompt, lower temperature, or add direct source constraints.

### #3
- **Question:** Tờ khai thuế GTGT trong BCTC.pdf sử dụng mẫu số nào?
- **Expected:** Mẫu số 01/GTGT.
- **Got:** Không tìm thấy.
- **Worst metric:** answer_relevancy
- **Error Tree:** Output sai → Context đúng? (Thông tin mẫu số nằm ở đầu trang) → Query OK
- **Root cause:** LLM hallucinating. Metadata/Header information might be lost during chunking or not prioritized by the model.
- **Suggested fix:** Tighten prompt, lower temperature, or add direct source constraints.

### #4
- **Question:** Nghị định 13/2023/NĐ-CP có quy định một mức phạt tiền cụ thể cho hành vi vi phạm bảo vệ dữ liệu cá nhân không?
- **Expected:** Không. Tài liệu chỉ nêu rằng tùy mức độ vi phạm mà tổ chức, cá nhân có thể bị xử lý kỷ luật, xử phạt vi phạm hành chính hoặc xử lý hình sự.
- **Got:** Không tìm thấy.
- **Worst metric:** faithfulness
- **Error Tree:** Output sai → Context đúng? (Văn bản pháp luật mô tả hình thức xử lý) → Query OK
- **Root cause:** LLM hallucinating. The model failed to synthesize a qualitative "No" based on the descriptive text.
- **Suggested fix:** Tighten prompt, lower temperature, or add direct source constraints.

### #5
- **Question:** Kỳ tính thuế của tờ khai thuế GTGT là thời gian nào?
- **Expected:** Quý 4 năm 2024.
- **Got:** Không tìm thấy.
- **Worst metric:** faithfulness
- **Error Tree:** Output sai → Context đúng? (Trực tiếp có trong tờ khai) → Query OK
- **Root cause:** LLM hallucinating. Failure to map a specific temporal field from the retrieved chunks.
- **Suggested fix:** Tighten prompt, lower temperature, or add direct source constraints.

## Case Study (cho presentation)

**Question chọn phân tích:** "Tờ khai được lập ngày nào và người ký đại diện là ai?"

**Error Tree walkthrough:**
1. **Output đúng?** → Không (Got "Không tìm thấy" instead of specific details).
2. **Context đúng?** → Có (The context chunks include references to signing and representative roles in legal documents).
3. **Query rewrite OK?** → Có (Direct and specific question).
4. **Fix ở bước:** Generation (LLM failed to extract existing info from context).

**Nếu có thêm 1 giờ, sẽ optimize:**
- Refining the system prompt to explicitly handle form-style data extraction.
- Increasing the retrieval context window or using a different chunking strategy (like semantic chunking) to keep related entities (Date/Name) together.
