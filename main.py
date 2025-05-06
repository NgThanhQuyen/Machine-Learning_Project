import os
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np

from models import ModelFactory
from audio_utils import record_audio, recognize_speech, extract_features
from data_utils import prepare_data, evaluate_models, save_to_text
from visualization import plot_results

def main():
    # Chuẩn bị dữ liệu
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Tạo và huấn luyện các mô hình
    print("Đang huấn luyện các mô hình...")
    hmm_custom, hmm_lib, svm = ModelFactory.create_models()
    
    # Huấn luyện HMM tự cài đặt
    print("Huấn luyện HMM tự cài đặt...")
    hmm_custom.fit(X_train.reshape(-1, 1, X_train.shape[1]))
    y_pred1 = [hmm_custom.predict(x.reshape(1, 1, -1)) for x in X_test]
    
    # Huấn luyện HMM thư viện
    print("Huấn luyện HMM từ thư viện...")
    hmm_lib.fit(X_train)
    y_pred2 = hmm_lib.predict(X_test)
    
    # Huấn luyện SVM
    print("Huấn luyện SVM...")
    svm.fit(X_train, y_train)
    y_pred3 = svm.predict(X_test)
    # Đánh giá và vẽ biểu đồ
    metrics = evaluate_models(y_test, y_pred1, y_pred2, y_pred3)
    plot_results(metrics)
    
    # Demo: Ghi âm và nhận dạng
    while True:
        input("Nhấn Enter để bắt đầu ghi âm...")
        audio, temp_wav = record_audio()
        
        # Nhận dạng giọng nói thành văn bản
        speech_text = recognize_speech(temp_wav)
        print(f"\nVăn bản nhận dạng được: {speech_text}")
        
        # Nhận dạng bằng ML
        features = extract_features(audio)
        features_mean = np.mean(features, axis=0)
        
        pred1 = hmm_custom.predict(features_mean.reshape(1, 1, -1))
        pred2 = hmm_lib.predict(features_mean.reshape(1, -1))
        pred3 = svm.predict(features_mean.reshape(1, -1))
        
        # Lưu cả văn bản và kết quả ML vào file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_to_text(timestamp, speech_text, [pred1, pred2, pred3])
        
        # Xóa file WAV tạm
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        choice = input("\nTiếp tục? (y/n): ")
        if choice.lower() != 'y':
            break

if __name__ == "__main__":
    main()
