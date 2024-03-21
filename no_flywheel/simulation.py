import numpy as np
from pyquaternion import Quaternion

from src.orbit import Orbit
from src.rotation import Rotation
from src.control import Control
from src.first_integrals import First_integrals


class Simulation:
    t0 = 0
    y0 = None

    def __init__(self, orbit: Orbit, rotation: Rotation, control: Control, first_integrals: First_integrals):
        self.orbit = orbit
        self.rotation = rotation
        self.control = control
        self.first_integrals = first_integrals

        # добавляем начальные данные
        y0 = self.orbit.from_elements_to_xyz(a=orbit.a, e=orbit.e, i=orbit.i, w=orbit.w, omega=orbit.omega, nu=orbit.nu)
        self.y0 = np.concatenate((y0, rotation.W0, rotation.Q0.elements))



    def vector_function_right_parts(self, t: float, y: np.ndarray):
        """
        Рассчитывает правые части векторной функции для задачи двух тел

            :param t: Время (секунды).
            :param y: NumPy массив, содержащий текущие значения вектора r (y[0:3]) и вектора v (y[3:6]),
             w (y[6:9]), Q (y[9:13]).

        :return: NumPy массив, содержащий правые части дифференциальных уравнений задачи двух тел
        """
        right_parts = np.zeros(13)

        right_parts[0:3] = y[3:6]
        right_parts[3:6] = - self.orbit.mu * y[0:3] / (np.linalg.norm(y[0:3])) ** 3

        Q = Quaternion(y[9:13])
        r_ssk = Q.conjugate.rotate(y[0:3])

        M_grav = 3 * self.orbit.mu / (np.linalg.norm(r_ssk) ** 5) * np.cross(r_ssk, self.rotation.J @ r_ssk)
        U = self.control.get(t=t, y=y, Mext=M_grav)

        right_parts[6:9] = np.linalg.inv(self.rotation.J) @ (-np.cross(y[6:9], self.rotation.J @ y[6:9]) + M_grav + U)
        right_parts[9:13] = np.array(
            (0.5 * Q * Quaternion(
                vector=y[6:9])
             ).elements)

        return right_parts

    def runge_kutta_4(self, f, t0: float, y0: np.ndarray, h, n: int):
        """
        Метод Рунге-Кутта 4-го порядка для решения системы дифференциальных уравнений

        Аргументы:
        f: функция, описывающая систему дифференциальных уравнений
           Принимает аргументы t и y и возвращает массив значений производных dy/dx
        t0: начальное значение t
        y0: начальное значение y (массив значений)
        h: шаг интегрирования
        n: количество шагов
        """

        # Создаем массивы для хранения результатов
        t = np.zeros(n + 1)
        y = np.zeros((n + 1, len(y0)))

        # Записываем начальные значения
        t[0] = t0
        y[0] = y0

        # Итерационный процесс метода Рунге-Кутта 4-го порядка
        for i in range(n):
            k1 = f(t[i], y[i])
            k2 = f(t[i] + h / 2, y[i] + h / 2 * k1)
            k3 = f(t[i] + h / 2, y[i] + h / 2 * k2)
            k4 = f(t[i] + h, y[i] + h * k3)

            t[i + 1] = t[i] + h
            y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6

            # нормировка кватерниона в виде
            y[i + 1][9:13] = np.array(
                Quaternion(y[i + 1][9:13]).normalised.elements
            )
        return t, y


    def start(self, h: float, n: int):
        t, y = self.runge_kutta_4(f=self.vector_function_right_parts, t0=self.t0, y0=self.y0, h=h, n=n)

        # добавляем результаты по своим классам
        self.orbit.add_results(t=t, y=y[:, 0:6])
        self.rotation.add_results(t=t, y=y[:, 6:13])

        # Считаем первые интегралы
        self.first_integrals.calculate(t=t, y=y)


