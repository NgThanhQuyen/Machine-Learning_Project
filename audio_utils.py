import sounddevice as sd
import wave
import speech_recognition as sr
from python_speech_features import mfcc
import numpy as np

# Cài đặt các tham số
SAMPLE_RATE = 16000
DURATION = 2  # seconds
MFCC_FEATURES = 13

def record_audio():
    """
    Ghi âm từ microphone và lưu thành file WAV
    """
    print("Bắt đầu ghi âm...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), 
                      samplerate=SAMPLE_RATE, 
                      channels=1)
    sd.wait()
    print("Ghi âm kết thúc")
    
    # Lưu file WAV tạm thời
    temp_wav = "temp_recording.wav"
    with wave.open(temp_wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((recording * 32767).astype(np.int16).tobytes())
    
    return recording.flatten(), temp_wav

def recognize_speech(audio_file):
    """
    Nhận dạng giọng nói thành văn bản
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

def extract_features(audio):
    """
    Trích xuất đặc trưng MFCC từ tín hiệu âm thanh
    """
    mfcc_features = mfcc(audio, 
                        samplerate=SAMPLE_RATE, 
                        numcep=MFCC_FEATURES)
    return mfcc_features