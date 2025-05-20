import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from audio_utils import SAMPLE_RATE, extract_features

def prepare_data():
    """
    Chuẩn bị dữ liệu huấn luyện (giả lập)
    """
    # Sử dụng độ dài mẫu cố định cho dữ liệu huấn luyện (2 giây)
    TRAINING_DURATION = 2
    X = []
    y = []
    for i in range(100):
        audio = np.random.randn(SAMPLE_RATE * TRAINING_DURATION)
        features = extract_features(audio)
        # Lấy giá trị trung bình của các đặc trưng
        features_mean = np.mean(features, axis=0)
        X.append(features_mean)
        y.append(i % 2)  # 2 lớp: 0 và 1
    return np.array(X), np.array(y)

def evaluate_models(y_true, y_pred1, y_pred2, y_pred3):
    """
    Đánh giá hiệu suất của các mô hình bằng nhiều độ đo khác nhau.
    
    Các độ đo được sử dụng:    
    1. Accuracy (Độ chính xác):
       - Tỷ lệ dự đoán đúng trên tổng số mẫu
       - Công thức: (TP + TN) / (TP + TN + FP + FN)
       Trong đó:
       + TP (True Positive): Số mẫu "có tiếng nói" được dự đoán đúng là "có tiếng nói"
       + TN (True Negative): Số mẫu "không có tiếng nói" được dự đoán đúng là "không có tiếng nói"
       + FP (False Positive): Số mẫu "không có tiếng nói" bị dự đoán sai là "có tiếng nói"
       + FN (False Negative): Số mẫu "có tiếng nói" bị dự đoán sai là "không có tiếng nói"
    2. F1-score (Điểm F1):
       - Trung bình điều hòa của precision và recall
       - Phù hợp với dữ liệu mất cân bằng
       - Công thức: 2 * (precision * recall) / (precision + recall)
    
    3. Precision (Độ chính xác dương tính):
       - Tỷ lệ dự đoán đúng trong các dự đoán dương tính
       - Công thức: TP / (TP + FP)
    
    4. Recall (Độ nhạy):
       - Tỷ lệ phát hiện đúng các trường hợp dương tính thực
       - Công thức: TP / (TP + FN)
    
    Tham số:
        y_true (array): Nhãn thực tế
        y_pred1 (array): Dự đoán của HMM tự cài đặt
        y_pred2 (array): Dự đoán của HMM thư viện
        y_pred3 (array): Dự đoán của SVM
    
    Trả về:
        dict: Dictionary chứa các độ đo cho từng mô hình
    """
    metrics = {
        'accuracy': [],
        'f1_score': [],
        'precision': [],
        'recall': []
    }
    
    for y_pred in [y_pred1, y_pred2, y_pred3]:
        # Chuyển đổi các dự đoán thành mảng numpy
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        
        # Tính toán các độ đo
        metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        metrics['f1_score'].append(f1_score(y_true, y_pred, average='weighted'))
        metrics['precision'].append(precision_score(y_true, y_pred, average='weighted'))
        metrics['recall'].append(recall_score(y_true, y_pred, average='weighted'))
    
    # In kết quả chi tiết
    print("\nKết quả đánh giá chi tiết:")
    
    print("\nHMM tự cài đặt:")
    print(f"  Accuracy: {metrics['accuracy'][0]:.3f}")
    print(f"  F1-score: {metrics['f1_score'][0]:.3f}")
    print(f"  Precision: {metrics['precision'][0]:.3f}")
    print(f"  Recall: {metrics['recall'][0]:.3f}")
    
    print("\nHMM thư viện:")
    print(f"  Accuracy: {metrics['accuracy'][1]:.3f}")
    print(f"  F1-score: {metrics['f1_score'][1]:.3f}")
    print(f"  Precision: {metrics['precision'][1]:.3f}")
    print(f"  Recall: {metrics['recall'][1]:.3f}")
    
    print("\nSVM:")
    print(f"  Accuracy: {metrics['accuracy'][2]:.3f}")
    print(f"  F1-score: {metrics['f1_score'][2]:.3f}")
    print(f"  Precision: {metrics['precision'][2]:.3f}")
    print(f"  Recall: {metrics['recall'][2]:.3f}")
    
    return metrics

def save_to_text(timestamp, speech_text, ml_predictions, filename="speech_output.txt"):
    """
    Lưu kết quả nhận dạng và dự đoán vào file text.
    
    Định dạng lưu trữ:
    - Thời gian ghi âm (timestamp)
    - Văn bản được nhận dạng
    - Kết quả dự đoán của 3 mô hình:
      + HMM tự cài đặt: 0 (không có tiếng nói) / 1 (có tiếng nói)
      + HMM thư viện: 0 (không có tiếng nói) / 1 (có tiếng nói)
      + SVM: 0 (không có tiếng nói) / 1 (có tiếng nói)
    
    Tham số:
        timestamp (str): Thời gian ghi âm
        speech_text (str): Văn bản được nhận dạng
        ml_predictions (list): Kết quả dự đoán của 3 mô hình
        filename (str): Tên file để lưu kết quả
    """
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n--- Ghi âm lúc: {timestamp} ---\n")
        f.write(f"Nội dung nói: {speech_text}\n")
        f.write(f"Kết quả nhận dạng ML:\n")
        f.write(f"- HMM tự cài đặt: Trạng thái {ml_predictions[0]} ({'Có tiếng nói' if ml_predictions[0] == 1 else 'Không có tiếng nói'})\n")
        f.write(f"- HMM thư viện: Trạng thái {ml_predictions[1]} ({'Có tiếng nói' if ml_predictions[1] == 1 else 'Không có tiếng nói'})\n")
        f.write(f"- SVM: Trạng thái {ml_predictions[2]} ({'Có tiếng nói' if ml_predictions[2] == 1 else 'Không có tiếng nói'})\n")
        f.write("-" * 40 + "\n")
    print(f"Đã lưu kết quả vào file {filename}")