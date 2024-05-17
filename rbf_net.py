import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import scipy
import matplotlib.pyplot as plt


class Rbf_net:
    def __init__(self, master: list, slave: list, n_centers: int) -> None:
        self.data = slave
        self.data_2 = master

        self.n_centers = n_centers
        self.centers = None
        # Радиус
        self.sigma = None
        # Коэфициент C
        self.model_coef = None
        # матрицы союственных значений
        self.phi_matrix = None
        self.phi_matrix_cup = None
        # восстановленные данные
        self.reconstructed_data = None
        self.reconstructed_data_cup = None
        # Ошибки
        self.err_solo = None
        self.err_cup = None

    def _gauss(self, data: float, centers: list, sigma: list) -> float:
        return np.exp(-((np.linalg.norm(data - centers) ** 2) / (2 * (sigma**2))))

    def _centers(self) -> list:
        means = KMeans(n_clusters=self.n_centers, max_iter=100, random_state=3, n_init=5)
        means.fit(self.data)
        self.centers = means.cluster_centers_
        return self.centers

    def _sigma(self):
        self._centers()

        L = len(self.centers)
        self.sigma = np.empty(L)
        _dist = 0

        for i in range(L):
            node_neighbours = []
            for j in range(i, L):
                _dist = np.linalg.norm(self.centers[i] - self.centers[j])
                if _dist != 0:
                    node_neighbours.append(_dist)

            if node_neighbours:
                self.sigma[i] = min(node_neighbours) * 10
            self.sigma[-1] = self.sigma[-2]

        return self.sigma

    def _Phi_matrix(self, tau):
        self._sigma()
        N = len(self.data) - tau
        K = len(self.centers)

        self.phi_matrix = np.empty((N, K))

        for n in range(N):
            for k in range(K):
                self.phi_matrix[n][k] = self._gauss(
                    self.data[n], self.centers[k], self.sigma[k]
                )

        return self.phi_matrix

    def _Phi_matrix_cup(self, tau):
        N = len(self.data_2) - tau
        K = len(self.centers)

        self.phi_matrix_cup = np.empty((N, K))

        for n in range(N):
            for k in range(K):
                self.phi_matrix_cup[n][k] = self._gauss(
                    self.data_2[n], self.centers[k], self.sigma[k]
                )

        return self.phi_matrix_cup

    def solo_pred(self, tau):
        self._Phi_matrix(tau)

        self.model_coef = np.linalg.lstsq(self.phi_matrix, self.data[tau:], rcond=None)[0]
        
        self.reconstructed_data = self.phi_matrix @ self.model_coef

        self.err_solo = (
            np.sqrt(metrics.mean_squared_error(self.data[tau:], self.reconstructed_data))
        )

        return self.reconstructed_data

    def couple_pred(self, tau):
        self._Phi_matrix_cup(tau)

        phi_matrix_cup = np.hstack((self.phi_matrix, self.phi_matrix_cup))

        self.model_coef = np.linalg.lstsq(phi_matrix_cup, self.data[tau:], rcond=None)[0]
                                                                            
        self.reconstructed_data_cup = phi_matrix_cup @ self.model_coef

        self.err_cup = (
            np.sqrt(metrics.mean_squared_error(self.data[tau:], self.reconstructed_data_cup))
        )

        return self.reconstructed_data_cup
