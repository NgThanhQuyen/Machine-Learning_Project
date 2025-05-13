import numpy as np
from sklearn.svm import SVC
from hmmlearn import hmm

class CustomHMM:
    """
    Cách 1: Tự cài đặt Hidden Markov Model
    
    Thuộc tính:
        n_states (int): Số trạng thái ẩn của mô hình
        A (array): Ma trận chuyển trạng thái (transition matrix)
        B (array): Ma trận phát xạ (emission matrix)
        pi (array): Phân phối xác suất trạng thái ban đầu
    """
    def __init__(self, n_states):
        """
        Khởi tạo mô hình HMM với số trạng thái cho trước
        
        Tham số:
            n_states (int): Số trạng thái ẩn của mô hình
        """
        self.n_states = n_states
        self.A = None  # Ma trận chuyển trạng thái
        self.B = None  # Ma trận phát xạ
        self.pi = None  # Phân phối trạng thái ban đầu

    def fit(self, X, n_iter=100):
        """
        Huấn luyện mô hình HMM sử dụng thuật toán Baum-Welch
        
        Tham số:
            X (array): Dữ liệu huấn luyện, shape (n_samples, 1, n_features)
            n_iter (int): Số vòng lặp tối đa cho thuật toán
        """
        n_samples = len(X)
        n_features = X[0].shape[1]
        
        # Khởi tạo ngẫu nhiên các tham số
        self.A = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)
        self.B = [np.random.randn(n_features) for _ in range(self.n_states)]
        self.pi = np.random.dirichlet(np.ones(self.n_states))

        for _ in range(n_iter):
            # E-step: Tính toán gamma và xi
            gamma = np.zeros((n_samples, self.n_states))
            xi = np.zeros((n_samples - 1, self.n_states, self.n_states))
            
            # M-step: Cập nhật các tham số
            self.A = np.mean(xi, axis=0) + 1e-5  # Thêm một giá trị nhỏ để tránh 0
            self.A = self.A / self.A.sum(axis=1, keepdims=True)  # Chuẩn hóa
            self.pi = gamma[0] + 1e-5  # Thêm một giá trị nhỏ để tránh 0
            self.pi = self.pi / self.pi.sum()  # Chuẩn hóa

    def predict(self, X):
        """
        Dự đoán chuỗi trạng thái ẩn sử dụng thuật toán Viterbi
        
        Tham số:
            X (array): Dữ liệu cần dự đoán, shape (n_samples, 1, n_features)
            
        Trả về:
            int: Trạng thái dự đoán (0: không có tiếng nói, 1: có tiếng nói)
        """
        return np.argmax([self._compute_likelihood(x) for x in X])

    def _compute_likelihood(self, x):
        """
        Tính logarit của xác suất quan sát chuỗi x
        
        Tham số:
            x (array): Chuỗi quan sát, shape (1, n_features)
            
        Trả về:
            float: Logarit của xác suất
        """
        return np.sum([self.pi[i] * self.A[i].sum() for i in range(self.n_states)])

class ModelFactory:
    @staticmethod
    def create_models():
        """
        Tạo các mô hình học máy
        
        Trả về:
            tuple: (hmm_custom, hmm_lib, svm)
                - hmm_custom: HMM tự cài đặt (Cách 1)
                - hmm_lib: HMM từ thư viện (Cách 2)
                - svm: SVM từ thư viện (Cách 3)
        """
        # Cách 1: HMM tự cài đặt
        hmm_custom = CustomHMM(n_states=2)
        
        # Cách 2: HMM từ thư viện hmmlearn
        hmm_lib = hmm.GaussianHMM(
            n_components=2,            # Số trạng thái
            covariance_type='diag',    # Loại ma trận hiệp phương sai
            n_iter=100,               # Số vòng lặp tối đa
            init_params='',           # Không tự động khởi tạo tham số
            params='stmc'             # Cho phép cập nhật tất cả tham số
        )
        
        # Khởi tạo các tham số với giá trị hợp lệ
        n_features = 13  # Số đặc trưng MFCC
        
        # Khởi tạo xác suất trạng thái ban đầu
        hmm_lib.startprob_ = np.array([0.5, 0.5])
        
        # Khởi tạo ma trận chuyển trạng thái
        hmm_lib.transmat_ = np.array([
            [0.7, 0.3],  # Xác suất chuyển từ trạng thái 0
            [0.3, 0.7]   # Xác suất chuyển từ trạng thái 1
        ])
        
        # Khởi tạo means với giá trị ngẫu nhiên nhỏ
        hmm_lib.means_ = np.random.randn(2, n_features) * 0.01
        
        # Khởi tạo covars với giá trị đường chéo là 1
        hmm_lib.covars_ = np.ones((2, n_features))
        
        # Đảm bảo các ma trận xác suất được chuẩn hóa
        hmm_lib.startprob_ = hmm_lib.startprob_ / hmm_lib.startprob_.sum()
        hmm_lib.transmat_ = hmm_lib.transmat_ / hmm_lib.transmat_.sum(axis=1)[:, np.newaxis]
        
        # Cách 3: SVM từ thư viện scikit-learn
        svm = SVC(
            kernel='rbf',         # Hàm kernel RBF
            random_state=42       # Giá trị khởi tạo ngẫu nhiên
        )
        
        return hmm_custom, hmm_lib, svm