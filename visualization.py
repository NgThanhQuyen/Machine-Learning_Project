import matplotlib.pyplot as plt
import os
import numpy as np

def plot_results(metrics, timestamp):
    """
    Trực quan hóa kết quả đánh giá của các mô hình bằng biểu đồ cột.
    
    Tạo 4 biểu đồ con cho 4 độ đo:
    1. Accuracy (Độ chính xác):
       - Đánh giá tổng thể khả năng dự đoán đúng
       - Giá trị từ 0 đến 1 (càng cao càng tốt)
    
    2. F1-score (Điểm F1):
       - Cân bằng giữa precision và recall
       - Phù hợp với dữ liệu mất cân bằng
       - Giá trị từ 0 đến 1 (càng cao càng tốt)
    
    3. Precision (Độ chính xác dương tính):
       - Đánh giá khả năng dự đoán đúng các mẫu dương tính
       - Giá trị từ 0 đến 1 (càng cao càng tốt)
    
    4. Recall (Độ nhạy):
       - Đánh giá khả năng phát hiện đúng các mẫu dương tính
       - Giá trị từ 0 đến 1 (càng cao càng tốt)
    
    Tham số:
        metrics (dict): Dictionary chứa các độ đo cho từng mô hình
        timestamp (str): Thời gian để đặt tên file kết quả
    """
    methods = ['HMM\n(Tự cài đặt)', 'HMM\n(Thư viện)', 'SVM']
    measures = ['Accuracy', 'F1-score', 'Precision', 'Recall']
    
    # Tạo figure với 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Đánh giá hiệu suất các mô hình', fontsize=16)
    
    # Vẽ từng độ đo trên một biểu đồ con
    for idx, (measure, values) in enumerate([
        ('accuracy', metrics['accuracy']),
        ('f1_score', metrics['f1_score']),
        ('precision', metrics['precision']),
        ('recall', metrics['recall'])
    ]):
        row = idx // 2
        col = idx % 2
        
        ax = axes[row, col]
        bars = ax.bar(methods, values)
        
        # Thêm giá trị lên đầu mỗi cột
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
        
        ax.set_title(measures[idx])
        ax.set_ylim([0, 1])  # Các độ đo đều có giá trị từ 0 đến 1
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Tạo thư mục plots nếu chưa tồn tại
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Lưu biểu đồ với timestamp
    filename = os.path.join(plots_dir, f'results_{timestamp}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nĐã lưu biểu đồ kết quả vào: {filename}")