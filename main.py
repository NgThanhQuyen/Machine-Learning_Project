import os
from datetime import datetime
import numpy as np

from models import ModelFactory
from audio_utils import record_audio, recognize_speech, extract_features
from data_utils import prepare_data, evaluate_models, save_to_text
from visualization import plot_results

def create_training_data(features):
    """
    Tạo dữ liệu huấn luyện từ đặc trưng MFCC thực tế
    """
    # Tính toán ngưỡng năng lượng để phân biệt giọng nói và im lặng
    energy = np.sum(features ** 2, axis=1)
    threshold = np.mean(energy) + 0.5 * np.std(energy)
    
    # Gán nhãn dựa trên năng lượng
    labels = (energy > threshold).astype(int)
    
    # Tạo chuỗi dữ liệu có cấu trúc thời gian
    X = []
    y = []
    
    # Sử dụng cửa sổ trượt để tạo mẫu
    window_size = 5
    for i in range(len(features) - window_size + 1):
        window_features = features[i:i+window_size]
        window_label = labels[i:i+window_size]
        
        # Tính đặc trưng trung bình của cửa sổ
        X.append(np.mean(window_features, axis=0))
        # Gán nhãn là 1 nếu phần lớn khung thời gian có giọng nói
        y.append(1 if np.mean(window_label) > 0.5 else 0)
    
    X = np.array(X)
    y = np.array(y)
    
    # Thêm nhiễu nhỏ để tăng tính đa dạng
    noise = np.random.normal(0, 0.001, X.shape)
    X = X + noise
    
    # Chuẩn hóa dữ liệu
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
    
    return X, y

def main():
    # Tạo và huấn luyện các mô hình
    print("Khởi tạo các mô hình...")
    hmm_custom, hmm_lib, svm = ModelFactory.create_models()
    
    # Demo: Ghi âm và nhận dạng
    while True:
        input("Nhấn Enter để bắt đầu ghi âm...")
        audio, temp_wav = record_audio()
        
        # Nhận dạng giọng nói thành văn bản
        speech_text = recognize_speech(temp_wav)
        print(f"\nVăn bản nhận dạng được: {speech_text}")
        
        try:
            # Trích xuất đặc trưng MFCC
            features = extract_features(audio)
            
            # Kiểm tra tính hợp lệ của features
            if len(features) == 0 or np.any(np.isnan(features)) or np.any(np.isinf(features)):
                raise ValueError("Không thể trích xuất đặc trưng hợp lệ từ âm thanh")
            
            print(f"\nĐã trích xuất {len(features)} khung thời gian, mỗi khung có {features.shape[1]} đặc trưng MFCC")
            
            # Tạo dữ liệu huấn luyện từ đặc trưng MFCC
            X, y = create_training_data(features)
            print(f"Tạo được {len(X)} mẫu huấn luyện")
            
            # Kiểm tra phân bố của nhãn
            n_speech = np.sum(y == 1)
            n_silence = np.sum(y == 0)
            print(f"Phân bố nhãn: {n_speech} mẫu có tiếng nói, {n_silence} mẫu không có tiếng nói")
            
            # Chia dữ liệu huấn luyện và kiểm tra
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Huấn luyện các mô hình
            print("\nĐang huấn luyện các mô hình...")
            
            print("1. Huấn luyện HMM tự cài đặt...")
            hmm_custom.fit(X_train.reshape(-1, 1, X_train.shape[1]))
            
            print("2. Huấn luyện HMM thư viện...")
            hmm_lib.fit(X_train)
            
            print("3. Huấn luyện SVM...")
            svm.fit(X_train, y_train)
            
            # Dự đoán và đánh giá
            print("\nĐang đánh giá các mô hình...")
            y_pred1 = [hmm_custom.predict(x.reshape(1, 1, -1)) for x in X_test]
            y_pred2 = hmm_lib.predict(X_test)
            y_pred3 = svm.predict(X_test)
            
            # Đánh giá chi tiết và vẽ biểu đồ
            metrics = evaluate_models(y_test, y_pred1, y_pred2, y_pred3)
            
            # Lấy timestamp và lưu kết quả
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plot_results(metrics, timestamp)
            
            # Dự đoán trạng thái cuối cùng
            features_mean = np.mean(features, axis=0).reshape(1, -1)
            pred1 = hmm_custom.predict(features_mean.reshape(1, 1, -1))
            pred2 = hmm_lib.predict(features_mean)
            pred3 = svm.predict(features_mean)
            
            save_to_text(timestamp, speech_text, [pred1, pred2, pred3])
            
        except Exception as e:
            print(f"\nLỗi trong quá trình xử lý: {str(e)}")
            print("Đang tiếp tục...")
        
        # Xóa file WAV tạm
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        choice = input("\nTiếp tục? (y/n): ")
        if choice.lower() != 'y':
            break

if __name__ == "__main__":
    main()
