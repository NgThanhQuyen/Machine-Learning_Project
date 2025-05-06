# Dự án Nhận dạng Giọng nói sử dụng Machine Learning

## Giới thiệu
Dự án này là một hệ thống nhận dạng giọng nói tích hợp nhiều phương pháp Machine Learning để so sánh hiệu suất. Hệ thống có khả năng ghi âm giọng nói thời gian thực, chuyển đổi thành văn bản và phân loại bằng các mô hình ML.

### Các tính năng chính
- Ghi âm giọng nói trực tiếp từ microphone
- Chuyển đổi giọng nói thành văn bản sử dụng Google Speech API
- Trích xuất đặc trưng MFCC từ tín hiệu âm thanh
- So sánh 3 mô hình học máy khác nhau
- Trực quan hóa kết quả bằng biểu đồ

## Cấu trúc dự án

### 1. main.py
File chính điều phối toàn bộ luồng hoạt động của hệ thống:
- Khởi tạo và huấn luyện các mô hình ML
- Xử lý luồng ghi âm realtime
- Tích hợp các module xử lý âm thanh và nhận dạng
- Quản lý quá trình lưu trữ và hiển thị kết quả

### 2. models.py
Định nghĩa các mô hình học máy:
- CustomHMM: Mô hình Hidden Markov tự cài đặt
  + Sử dụng thuật toán Baum-Welch để huấn luyện
  + Thuật toán Viterbi cho dự đoán
- HMM từ thư viện hmmlearn: Triển khai chuẩn của HMM
- SVM: Sử dụng cho phân loại nhị phân
- ModelFactory: Quản lý việc tạo và khởi tạo các mô hình

### 3. audio_utils.py
Xử lý các tác vụ liên quan đến âm thanh:
- Ghi âm từ microphone (thời lượng 2 giây, tần số lấy mẫu 16kHz)
- Chuyển đổi giọng nói thành văn bản (hỗ trợ tiếng Việt)
- Trích xuất 13 đặc trưng MFCC từ tín hiệu âm thanh
- Xử lý và lưu trữ file WAV tạm thời

### 4. data_utils.py
Xử lý và quản lý dữ liệu:
- Tạo và chuẩn bị dữ liệu huấn luyện
- Tính toán các độ đo đánh giá:
  + Accuracy (độ chính xác)
  + F1-score (điểm F1)
- Lưu kết quả nhận dạng vào file với timestamp

### 5. visualization.py
Trực quan hóa kết quả:
- Vẽ biểu đồ so sánh song song các mô hình
- Hiển thị 2 độ đo: accuracy và F1-score
- Tự động lưu biểu đồ dưới dạng file PNG

## Yêu cầu hệ thống

### Môi trường
- Python 3.6+
- Microphone hoạt động
- Kết nối Internet (cho Google Speech API)

### Thư viện Python
Xem chi tiết trong file requirements.txt:
- numpy, scipy: Xử lý số liệu và tính toán
- scikit-learn, hmmlearn: Các mô hình ML
- librosa, sounddevice: Xử lý âm thanh
- python-speech-features: Trích xuất MFCC
- SpeechRecognition: API nhận dạng giọng nói
- matplotlib, pandas: Trực quan hóa và xử lý dữ liệu

## Cài đặt và Sử dụng

### 1. Cài đặt môi trường
```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Chạy chương trình
```bash
python main.py
```

### 3. Hướng dẫn sử dụng
1. Khởi động chương trình
2. Nhấn Enter để bắt đầu ghi âm
3. Nói trong 2 giây (đợi thông báo kết thúc)
4. Xem kết quả:
   - Văn bản được nhận dạng
   - Dự đoán của 3 mô hình
   - Biểu đồ so sánh (nếu có)
5. Chọn:
   - 'y' để tiếp tục ghi âm mới
   - 'n' để kết thúc chương trình

## Kết quả và File xuất ra

### 1. speech_output.txt
- Lưu lịch sử các lần ghi âm
- Định dạng cho mỗi bản ghi:
  + Thời gian ghi âm
  + Nội dung nhận dạng được
  + Kết quả dự đoán của 3 mô hình
    - HMM tự cài đặt:
        + 0: Không nhận dạng được âm thanh(không có tiếng nói)
        + 1: Nhận dạng được âm thanh(có tiếng nói)
    - HMM thư viện:
        + [0]: Không nhận dạng được âm thanh(không có tiếng nói)
        + [1]: Nhận dạng được âm thanh(có tiếng nói)
    - SVM:
        + [0]: Không nhận dạng được âm thanh(không có tiếng nói)
        + [1]: Nhận dạng được âm thanh(có tiếng nói)

### 2. results.png
- Biểu đồ so sánh hiệu suất
- Hai biểu đồ con:
  + Độ chính xác (Accuracy)
  + Điểm F1 (F1-score)

## Ghi chú
- Mô hình HMM tự cài đặt đơn giản hóa một số bước tính toán
- Google Speech API yêu cầu kết nối Internet
- File WAV tạm thời sẽ tự động bị xóa sau mỗi lần ghi âm
- Dữ liệu huấn luyện được tạo ngẫu nhiên cho mục đích demo