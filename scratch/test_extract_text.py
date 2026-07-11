import re

def extract_strings_from_doc(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    
    # 1. Thử trích xuất các chuỗi unicode (UTF-16LE) - định dạng Word thường dùng
    # Ở UTF-16LE, các ký tự Latin thường có byte thứ hai là 0x00
    # Ta tìm các chuỗi các ký tự in được kéo dài
    unicode_strings = []
    # Quy tắc: chuỗi byte có dạng: [char_code][0x00] liên tục
    # Tìm các đoạn bytes chẵn, lọc các chuỗi ký tự unicode
    # Word lưu text liên tục trong stream. Ta quét toàn bộ file tìm các ký tự unicode hợp lệ
    try:
        text = data.decode('utf-16le', errors='ignore')
        # Lọc bỏ các ký tự điều khiển rác, chỉ giữ lại chữ tiếng Việt, chữ thường, số, dấu câu
        # Tiếng Việt trong unicode bao gồm các block ký tự Latin mở rộng
        clean_text = re.sub(r'[^\x20-\x7E\u00C0-\u1EF9\n\t\r]', ' ', text)
        # Gom các khoảng trắng liên tiếp
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text
    except Exception as e:
        return f"Lỗi: {e}"

# Chạy thử với file đã down
text = extract_strings_from_doc("scratch/test_ctdl.doc")
print(f"Extracted length: {len(text)}")
print("First 1000 characters:")
print(text[:1000])
