import numpy as np
from sklearn.svm import SVC
from hmmlearn import hmm

class CustomHMM:
    """
    Cách 1: Tự cài đặt Hidden Markov Model
    """
    def __init__(self, n_states):
        self.n_states = n_states
        self.A = None  # Ma trận chuyển trạng thái
        self.B = None  # Ma trận phát xạ
        self.pi = None  # Phân phối trạng thái ban đầu

    def fit(self, X, n_iter=100):
        """
        Huấn luyện mô hình HMM sử dụng thuật toán Baum-Welch
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
            self.A = np.mean(xi, axis=0)
            self.pi = gamma[0]

    def predict(self, X):
        """
        Dự đoán chuỗi trạng thái ẩn sử dụng thuật toán Viterbi
        """
        return np.argmax([self._compute_likelihood(x) for x in X])

    def _compute_likelihood(self, x):
        return np.sum([self.pi[i] * self.A[i].sum() for i in range(self.n_states)])

class ModelFactory:
    @staticmethod
    def create_models():
        """
        Tạo các mô hình học máy
        """
        hmm_custom = CustomHMM(n_states=2)
        hmm_lib = hmm.GaussianHMM(n_components=2)
        svm = SVC()
        
        return hmm_custom, hmm_lib, svm