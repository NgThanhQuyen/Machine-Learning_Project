import sounddevice as sd
import wave
import speech_recognition as sr
from python_speech_features import mfcc
import numpy as np
import keyboard
import time

# Cài đặt các tham số
SAMPLE_RATE = 16000  # Tần số lấy mẫu (Hz)
MFCC_FEATURES = 13   # Số đặc trưng MFCC cần trích xuất

def record_audio():
    """
    Ghi âm từ microphone và lưu thành file WAV.
    Người dùng nhấn và giữ phím Space để ghi âm, thả ra để kết thúc.
    
    Trả về:
        tuple: (audio_data, temp_wav_path)
            - audio_data: Mảng numpy chứa dữ liệu âm thanh đã chuẩn hóa
            - temp_wav_path: Đường dẫn đến file WAV tạm thời
    """
    print("Nhấn và giữ SPACE để bắt đầu ghi âm, thả ra để kết thúc...")
    
    # Đợi người dùng nhấn phím Space
    keyboard.wait('space')
    print("Đang ghi âm... (thả phím SPACE để kết thúc)")
    
    # Bắt đầu ghi âm
    recording = sd.rec(int(SAMPLE_RATE * 10),  # Tối đa 10 giây
                      samplerate=SAMPLE_RATE,
                      channels=1,
                      blocking=False)
    
    # Đợi cho đến khi phím Space được thả ra
    start_time = time.time()
    while keyboard.is_pressed('space'):
        time.sleep(0.01)  # Giảm tải CPU
        if time.time() - start_time > 10:  # Giới hạn tối đa 10 giây
            break
    
    # Dừng ghi âm
    sd.stop()
    print("Ghi âm kết thúc")
    
    # Cắt phần âm thanh đã ghi được
    recorded_frames = int((time.time() - start_time) * SAMPLE_RATE)
    recording = recording[:recorded_frames]
    
    # Chuẩn hóa âm thanh về dải [-1, 1]
    if recording.size > 0:
        recording = np.clip(recording, -1.0, 1.0)
        # Chuyển đổi sang int16 an toàn
        int16_data = (recording * 32767).clip(-32768, 32767).astype(np.int16)
    else:
        # Nếu không có âm thanh, tạo mảng zeros
        int16_data = np.zeros(SAMPLE_RATE, dtype=np.int16)
    
    # Lưu file WAV tạm thời
    temp_wav = "temp_recording.wav"
    with wave.open(temp_wav, 'wb') as wf:
        wf.setnchannels(1)           # Mono
        wf.setsampwidth(2)           # 16-bit
        wf.setframerate(SAMPLE_RATE) # Tần số lấy mẫu
        wf.writeframes(int16_data.tobytes())
    
    return recording.flatten(), temp_wav

def recognize_speech(audio_file):
    """
    Nhận dạng giọng nói thành văn bản sử dụng Google Speech Recognition
    
    Tham số:
        audio_file (str): Đường dẫn đến file WAV cần nhận dạng
        
    Trả về:
        str: Văn bản được nhận dạng hoặc thông báo lỗi
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            # Thử nhận dạng bằng Google Speech Recognition
            text = recognizer.recognize_google(audio, language='vi-VN')
            return text
        except sr.UnknownValueError:
            return "Không thể nhận dạng giọng nói"
        except sr.RequestError:
            return "Lỗi kết nối đến dịch vụ nhận dạng giọng nói"
    
    # Trích xuất MFCC
    mfcc_features = mfcc(audio, 
                        samplerate=SAMPLE_RATE, 
                        numcep=MFCC_FEATURES)
    
    # Xử lý các giá trị không hợp lệ
    mfcc_features = np.nan_to_num(mfcc_features)
    
    return mfcc_features

def extract_features(audio):
    """
    Trích xuất đặc trưng MFCC (Mel Frequency Cepstral Coefficients) từ tín hiệu âm thanh.
    MFCC là đặc trưng quan trọng trong xử lý giọng nói, đại diện cho đặc tính của âm thanh
    trong miền tần số Mel, phù hợp với cách con người nghe âm thanh.
    
    Quy trình xử lý:
    1. Chuẩn hóa tín hiệu âm thanh đầu vào
    2. Loại bỏ các giá trị NaN/inf nếu có
    3. Trích xuất 13 đặc trưng MFCC cho mỗi khung thời gian
    4. Xử lý lại kết quả để đảm bảo tính hợp lệ
    
    Tham số:
        audio (array): Mảng 1D chứa dữ liệu âm thanh thô
        
    Trả về:
        array: Ma trận đặc trưng MFCC, shape (số_khung_thời_gian, 13)
               Mỗi hàng là một vector 13 đặc trưng MFCC cho một khung thời gian
    """
    # Đảm bảo audio không có giá trị NaN hoặc inf
    audio = np.nan_to_num(audio)
    
    # Chuẩn hóa âm thanh về dải [-1, 1]
    if np.std(audio) > 0:
        audio = (audio - np.mean(audio)) / np.std(audio)
    
    # Trích xuất MFCC với 13 hệ số
    mfcc_features = mfcc(audio, 
                        samplerate=SAMPLE_RATE,  # Tần số lấy mẫu
                        numcep=MFCC_FEATURES,    # Số hệ số MFCC cần trích xuất
                        nfilt=26,               # Số bộ lọc Mel
                        nfft=512)               # Kích thước cửa sổ FFT
    
    # Xử lý các giá trị không hợp lệ và chuẩn hóa
    mfcc_features = np.nan_to_num(mfcc_features)  # Thay thế NaN/inf
    
    return mfcc_features