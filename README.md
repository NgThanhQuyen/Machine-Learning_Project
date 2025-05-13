# Dự án Nhận dạng Giọng nói sử dụng Machine Learning

## Giới thiệu
Dự án này là một hệ thống nhận dạng giọng nói tích hợp nhiều phương pháp Machine Learning để phân loại và nhận dạng tiếng nói. Hệ thống có khả năng ghi âm giọng nói thời gian thực, chuyển đổi thành văn bản và phân loại bằng các mô hình ML.

### Các tính năng chính
- Ghi âm giọng nói trực tiếp từ microphone (tối đa 10 giây)
- Chuyển đổi giọng nói thành văn bản sử dụng Google Speech Recognition (hỗ trợ tiếng Việt)
- Trích xuất đặc trưng MFCC từ tín hiệu âm thanh
- So sánh hiệu suất của 3 mô hình học máy khác nhau
- Trực quan hóa kết quả với nhiều độ đo (Accuracy, F1-score, Precision, Recall)
- Lưu trữ kết quả và biểu đồ theo thời gian

## Cấu trúc dự án

### 1. main.py
File chính điều phối toàn bộ luồng hoạt động của hệ thống:
- Ghi âm và xử lý âm thanh realtime
- Tạo dữ liệu huấn luyện từ đặc trưng MFCC
- Huấn luyện và đánh giá các mô hình ML
- Quản lý quá trình lưu trữ kết quả

### 2. models.py
Định nghĩa các mô hình học máy:
- CustomHMM: Mô hình Hidden Markov tự cài đặt
  + Thuật toán Baum-Welch cho huấn luyện
  + Dự đoán dựa trên likelihood
- HMM từ thư viện hmmlearn
  + Sử dụng GaussianHMM với 2 trạng thái
  + Tham số được khởi tạo tối ưu
- SVM từ scikit-learn
  + Kernel RBF cho phân loại nhị phân
  + Tối ưu cho bài toán phân loại âm thanh

### 3. audio_utils.py
Xử lý các tác vụ liên quan đến âm thanh:
- Ghi âm từ microphone (sử dụng sounddevice)
- Xử lý và chuẩn hóa tín hiệu âm thanh
- Chuyển đổi giọng nói thành văn bản (Google Speech Recognition)
- Trích xuất và chuẩn hóa đặc trưng MFCC

### 4. data_utils.py
Xử lý và đánh giá dữ liệu:
- Đánh giá mô hình với nhiều độ đo:
  + Accuracy (độ chính xác)
  + F1-score (điểm F1)
  + Precision (độ chính xác dương tính)
  + Recall (độ nhạy)
- Lưu kết quả nhận dạng với timestamp

### 5. visualization.py
Trực quan hóa kết quả:
- Vẽ biểu đồ so sánh các mô hình
- Hiển thị 4 độ đo trên các biểu đồ con
- Tự động lưu biểu đồ với timestamp

## Cấu trúc thư mục
```
├── main.py              # File chính điều khiển chương trình
├── models.py            # Định nghĩa các mô hình ML
├── audio_utils.py       # Xử lý âm thanh và trích xuất đặc trưng
├── data_utils.py        # Xử lý dữ liệu và đánh giá
├── visualization.py     # Trực quan hóa kết quả
├── requirements.txt     # Danh sách thư viện cần thiết
├── speech_output.txt    # File lưu kết quả nhận dạng
└── plots/              # Thư mục chứa biểu đồ kết quả
    └── results_*.png   # Các file biểu đồ theo timestamp
```

## Yêu cầu hệ thống

### Môi trường
- Python 3.6+
- Microphone hoạt động
- Kết nối Internet (cho Google Speech Recognition)

### Thư viện Python
Xem chi tiết trong file requirements.txt:
- numpy, scipy: Xử lý số liệu
- scikit-learn, hmmlearn: Các mô hình ML
- sounddevice: Ghi âm realtime
- python-speech-features: Trích xuất MFCC
- SpeechRecognition: API nhận dạng giọng nói
- matplotlib: Trực quan hóa dữ liệu
- keyboard: Xử lý sự kiện bàn phím

## Cài đặt và Sử dụng

### 1. Cài đặt môi trường
```powershell
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
.\venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Chạy chương trình
```powershell
python main.py
```

### 3. Hướng dẫn sử dụng
1. Khởi động chương trình
2. Nhấn Enter để bắt đầu
3. Nhấn và giữ phím SPACE để ghi âm
4. Thả phím SPACE để kết thúc ghi âm
5. Xem kết quả:
   - Văn bản được nhận dạng
   - Kết quả dự đoán của các mô hình
   - Biểu đồ đánh giá hiệu suất
6. Chọn:
   - 'y' để tiếp tục ghi âm mới
   - 'n' để kết thúc chương trình

## Kết quả đầu ra

### 1. speech_output.txt
- Lưu kết quả mỗi lần ghi âm với:
  + Thời gian ghi âm (timestamp)
  + Văn bản nhận dạng được
  + Kết quả dự đoán của 3 mô hình:
    - HMM tự cài đặt: 0 (không có tiếng nói) / 1 (có tiếng nói)
    - HMM thư viện: 0 (không có tiếng nói) / 1 (có tiếng nói)
    - SVM: 0 (không có tiếng nói) / 1 (có tiếng nói)

### 2. Biểu đồ kết quả (plots/results_*.png)
- So sánh hiệu suất các mô hình với 4 độ đo:
  + Accuracy: Tỷ lệ dự đoán đúng
  + F1-score: Trung bình điều hòa của precision và recall
  + Precision: Tỷ lệ dự đoán đúng trong các dự đoán dương tính
  + Recall: Tỷ lệ phát hiện đúng các trường hợp dương tính thực