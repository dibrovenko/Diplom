import numpy as np
from pyquaternion import Quaternion


class First_integrals:

    result_c = result_E = result_f = result_h = None
    result_t = None

    def __init__(self, mu: float, J: np.ndarray):
        self.mu = mu  # Гравитационный параметр
        self.J = J  # Тензор инерции в главных осях.

    def calculate(self, t: np.ndarray, y: np.ndarray):
        """
        Рассчитывает три первых интреграла E, c, f и интеграл Якоби.
            :param y: NumPy массив, содержащий вектор состояния: y[:, 0:6] в декартовых координатах, y[:, 6:13] в ССК.
        """
        self.result_t = t
        r = y[:, 0:3]
        v = y[:, 3:6]
        W = y[:, 6:9]

        self.result_c = np.cross(r, v)
        self.result_E = (np.linalg.norm(v, axis=1) ** 2) / 2 - self.mu / np.linalg.norm(r, axis=1)
        self.result_f = np.cross(v, self.result_c) - self.mu * r / np.linalg.norm(r, axis=1)[:, np.newaxis]

        # Считаем интеграл Якоби в ССK
        self.result_h = np.zeros(len(r))

        for i in range(len(y)):
            Q = Quaternion(y[i][9:13])
            r_ssk = Q.conjugate.rotate(r[i])
            v_ssk = Q.conjugate.rotate(v[i])
            W0_ssk = np.cross(r_ssk, v_ssk) / (np.linalg.norm(r_ssk) ** 2)

            self.result_h[i] = 0.5 * np.dot(W[i], self.J @ W[i]) +\
                               1.5 * self.mu / (np.linalg.norm(r_ssk) ** 5) * np.dot(r_ssk, self.J @ r_ssk) -\
                               np.dot(W[i], self.J @ W0_ssk)
