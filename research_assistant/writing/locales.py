"""Vietnamese review translations and language-specific validation patterns."""

VI = {
    "title": "Tổng quan tài liệu",
    "status": "Trạng thái",
    "scope_caveat": "Bản tổng hợp chỉ mô tả snapshot đã trích xuất. Citation được kiểm tra cấu trúc; nội dung chưa được xác minh ngữ nghĩa. Các nhóm phương pháp dùng ngưỡng phân cụm tạm thời.",
    "template_caveat": "Phần template giữ nguyên văn evidence gốc, có thể khác ngôn ngữ được yêu cầu.",
    "input_warning": "Cảnh báo đầu vào: đã loại {count} evidence hướng dẫn định dạng tài liệu; xem JSON provenance.",
    "verification_title": "Phạm vi và điểm cần kiểm chứng",
    "verification_caveat": "Không suy ra 'chưa có nghiên cứu' từ trường dữ liệu còn thiếu. Dataset/metric là mentions, không chứng minh quan hệ đánh giá hay khả năng so sánh kết quả. Các hạn chế cần được kiểm chứng trước khi kết luận research gap.",
    "conclusion_title": "Kết luận",
    "conclusion": "Bản tổng hợp bao phủ evidence được chọn cho {count} tài liệu trong snapshot. Các nhóm phương pháp và hạn chế ở trên là cơ sở để đọc kiểm chứng; chưa xác lập tính đầy đủ của tổng quan hoặc tính mới của một hướng nghiên cứu.",
    "references": "Tài liệu tham khảo",
    "method_group": "Nhóm phương pháp {index}",
    "unassigned": "Chưa phân nhóm",
    "comparisons": "So sánh định tính",
    "gaps": "Hạn chế và hướng cần kiểm chứng"
}

VI_RANK_PATTERN = r"vượt trội|chưa (?:từng )?có nghiên cứu"
