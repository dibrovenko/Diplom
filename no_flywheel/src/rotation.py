import numpy as np
from pyquaternion import Quaternion


class Rotation:

    result_Wx = result_Wy = result_Wz = None
    result_Q = None
    result_t = None

    result_W_rel = []
    result_A = []

    def __init__(self, J: np.ndarray, Q0: Quaternion, W0: np.ndarray):
        """
        Конструктор класса, который инициализирует объект ориентации для углвого движения с заданными параметрами

            :param J: Тензор инерции в главных осях.
            :param Q0: Ориентация ССK относительно ИСК в нач момент времени.
            :param W0: Угловая относительная сокрость начальная в ССk.

        """
        self.J = J
        self.Q0 = Q0
        self.W0 = W0

    def add_results(self, t: np.ndarray, y: np.ndarray):
        """
        Добавляет результаты расчетов.
        """
        self.result_t = t
        self.result_Wx, self.result_Wy, self.result_Wz = y[:, 6:9].T
        self.result_Q = y[:, 9:13]

        self.result_W_rel = np.zeros((len(t), 3))
        self.result_A = np.zeros((len(t), 4))
        for i in range(len(t)):
            r = y[i][0:3]
            v = y[i][3:6]
            W_abs_cck = y[i][6:9]
            Q = Quaternion(y[i][9:13])

            W_ref_ock = np.array([0, np.linalg.norm(np.cross(r, v) / (np.linalg.norm(r) ** 2)), 0])

            # Ищем кватернион B (из ИСК в ОСК):
            e3 = r / np.linalg.norm(r)
            e2 = np.cross(r, v) / np.linalg.norm(np.cross(r, v))
            e1 = np.cross(e2, e3)
            K_matrix = np.array([e1, e2, e3])
            B = Quaternion(matrix=K_matrix).conjugate

            # Кватернион A (из ОСК в ССК) и Wrel в ssk:
            A = B.conjugate * Q
            W_rel_cck = W_abs_cck - A.conjugate.rotate(W_ref_ock)

            self.result_W_rel[i] = W_rel_cck
            self.result_A[i] = A.elements












