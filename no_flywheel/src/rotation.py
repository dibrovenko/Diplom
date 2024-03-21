import numpy as np
from pyquaternion import Quaternion


class Rotation:

    result_Wx = result_Wy = result_Wz = None
    result_Q = None
    result_t = None

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
        self.result_Wx, self.result_Wy, self.result_Wz = y[:, 0:3].T
        self.result_Q = y[:, 3:7]



