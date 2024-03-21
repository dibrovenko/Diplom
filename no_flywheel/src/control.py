import numpy as np
from pyquaternion import Quaternion


def quaternion_between_vectors(v1: np.ndarray, v2: np.ndarray):
    # Нормализация векторов
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Ось вращения
    axis = np.cross(v1, v2)

    # Угол поворота
    angle = np.arccos(np.dot(v1, v2))

    # Создание кватерниона
    q = Quaternion(axis=axis, angle=angle)

    return q



class Control:
    W_rel_isk_last = W_rel_isk_now = 0
    t_last = t_now = 0
    counter = 0
    U = np.array([0, 0, 0])

    def __init__(self, Kw: float, Kq: float, J: np.ndarray):
        """
        Конструктор класса, который инициализирует объект управления для углвого движения с заданными параметрами

            :param Kw:
            :param Kq:

        """
        self.Kw = Kw
        self.Kq = Kq
        self.J = J  # Тензор инерции в главных осях.


    def check_every_fourth_call(self):
        self.counter += 1
        if self.counter % 4 == 0:
            return True
        else:
            return False


    def get(self, t: float, y: np.ndarray, Mext: np.ndarray):
        """
        Рассчитывает управление

            :param t: Время (секунды).
            :param y: NumPy массив, содержащий текущие значения вектора r иск (y[0:3]) и вектора v иск (y[3:6]),
             w сск (y[6:9]), Q из иск в сск (y[9:13]).
            :param Mext: NumPy массив, содержащий внешний момент в сск.

        :return: NumPy массив, содержащий управления на Оси ССК
        """
        if not self.check_every_fourth_call():
            return self.U

        self.t_now = t
        r = y[0:3]
        v = y[3:6]
        W_abs_cck = y[6:9]
        Q = Quaternion(y[9:13])

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
        print(W_rel_cck)

        # считаем численно производную требуемой углвой скорости в сск
        self.W_rel_isk_now = np.cross(r, v) / (np.linalg.norm(r) ** 2)
        dotW_ref_ick = (self.W_rel_isk_now - self.W_rel_isk_last) / (self.t_now - self.t_last)
        dotW_ref_cck = Q.conjugate.rotate(dotW_ref_ick)
        self.t_last = t

        q = A.vector

        self.U = -Mext + np.cross(W_abs_cck, self.J @ W_abs_cck) - \
                 self.J @ np.cross(W_rel_cck, A.conjugate.rotate(W_ref_ock)) + self.J @ dotW_ref_cck - \
                 self.Kw * W_rel_cck + self.Kq * q

        return self.U








