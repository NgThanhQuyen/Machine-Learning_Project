import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from audio_utils import SAMPLE_RATE, DURATION, extract_features

def prepare_data():
    """
    Chuẩn bị dữ liệu huấn luyện (giả lập)
    """
    X = []
    y = []
    for i in range(100):
        audio = np.random.randn(SAMPLE_RATE * DURATION)
        features = extract_features(audio)
        # Lấy giá trị trung bình của các đặc trưng
        features_mean = np.mean(features, axis=0)
        X.append(features_mean)
        y.append(i % 2)  # 2 lớp: 0 và 1
    return np.array(X), np.array(y)

def evaluate_models(y_true, y_pred1, y_pred2, y_pred3):
    """
    Đánh giá hiệu suất của các mô hình
    """
    # Độ đo 1: Accuracy
    acc1 = accuracy_score(y_true, y_pred1)
    acc2 = accuracy_score(y_true, y_pred2)
    acc3 = accuracy_score(y_true, y_pred3)
    
    # Độ đo 2: F1-score
    f1_1 = f1_score(y_true, y_pred1, average='weighted')
    f1_2 = f1_score(y_true, y_pred2, average='weighted')
    f1_3 = f1_score(y_true, y_pred3, average='weighted')
    
    return {
        'accuracy': [acc1, acc2, acc3],
        'f1_score': [f1_1, f1_2, f1_3]
    }

def save_to_text(timestamp, speech_text, ml_predictions, filename="speech_output.txt"):
    """
    Lưu kết quả nhận dạng vào file text
    """
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n--- Ghi âm lúc: {timestamp} ---\n")
        f.write(f"Nội dung nói: {speech_text}\n")
        f.write(f"Kết quả nhận dạng ML:\n")
        f.write(f"- HMM tự cài đặt: {ml_predictions[0]}\n")
        f.write(f"- HMM thư viện: {ml_predictions[1]}\n")
        f.write(f"- SVM: {ml_predictions[2]}\n")
        f.write("-" * 40 + "\n")
    print(f"Đã lưu kết quả vào file {filename}")