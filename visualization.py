import matplotlib.pyplot as plt

def plot_results(metrics):
    """
    Trực quan hóa kết quả của các mô hình
    """
    methods = ['HMM (Tự cài đặt)', 'HMM (Thư viện)', 'SVM']
    
    # Biểu đồ độ chính xác
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(methods, metrics['accuracy'])
    plt.title('Độ chính xác của các phương pháp')
    plt.ylabel('Accuracy')
    
    # Biểu đồ F1-score
    plt.subplot(1, 2, 2)
    plt.bar(methods, metrics['f1_score'])
    plt.title('F1-score của các phương pháp')
    plt.ylabel('F1-score')
    
    plt.tight_layout()
    plt.savefig('results.png')
    plt.close()