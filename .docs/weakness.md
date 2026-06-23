## 1 Phaner 
# 1. Prompt hiện tại còn hơi đơn giản
Prompt hiện tại đã phân biệt RAG/GRAG, nhưng chưa thật sự có rubric mạnh cho:
khi nào tách task
khi nào không tách
khi nào cần hỏi lại
khi nào thiếu dữ liệu
Nên với câu khó, Planner có thể chưa ổn định.

# 2. Chưa có confidence score
Planner chưa trả:
confidence
reasoning_summary
needs_clarification
Vì vậy Executor không biết task này Planner có chắc không.

# 3. Chưa có cơ chế hỏi lại khi câu mơ hồ
Ví dụ:
Em đăng ký được môn đó chưa?
Planner có thể đoán đại thay vì trả:
Cần hỏi lại sinh viên môn nào.
Hiện tại schema chưa có task kiểu CLARIFY.

# 4. Không biết trạng thái dữ liệu thật
Planner không biết Neo4j có môn đó không, Qdrant có tài liệu đó không, hoặc sinh viên đã upload bảng điểm chưa. Nó chỉ lập kế hoạch dựa trên câu hỏi.
Việc kiểm tra dữ liệu thật nằm ở Executor/Worker.

# 5. Chưa giới hạn số task tối đa 
Nếu như người dùng cố tình hỏi câu hỏi cực dài thì hệ thống sẽ sử lý thế nào 

## 2. Executor
# 1. Chưa có retry cho task lỗi
Nếu RAG hoặc GRAG lỗi do tạm thời, hiện tại chỉ mark failed, chưa thử lại.

# 2. Error message còn khá chung
Ví dụ:
Không thể truy xuất thông tin phần RAG
Nó chưa nói rõ lỗi do:
Qdrant lỗi
Neo4j lỗi
OpenAI lỗi
missing transcript

# 3. GRAG vẫn phụ thuộc EntityExtractor
Với task GRAG, Executor có thể phải extract mã môn lại. Nếu extractor sai hoặc graph thiếu dữ liệu, GRAG trả kết quả yếu.

# 4. Chưa có cơ chế ưu tiên task
Hiện tại task nào cũng chạy như nhau. Nhưng trong thực tế, có task phụ thuộc task khác.
Ví dụ:
Kiểm tra môn tiên quyết trước
rồi mới kết luận đăng ký
Hiện tại hệ thống chạy song song, chưa có dependency graph.

# 5. Chưa có task cancellation
Nếu một task quá lâu, hiện chưa thấy timeout/cancel riêng cho từng task.

# 6. Trace chưa chuẩn hóa hoàn toàn với response cuối
Executor trả raw_data, còn các tầng sau phải diễn giải. Nếu chuẩn hóa thêm answer, evidence, sources, confidence, thì đánh giá sẽ tốt hơn