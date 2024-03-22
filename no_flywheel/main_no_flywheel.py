import numpy as np
from pyquaternion import Quaternion

from src.orbit import Orbit
from src.rotation import Rotation
from src.control import Control
from src.first_integrals import First_integrals
from simulation import Simulation
from grafics import start_plot

a = (6371 + 500) * 1000  # Большая полуось
e = 0.0  # Эксцентриситет
i = 0.9  # Наклонение
omega = 0.7  # Долгота восходящего узла
w = 1.37  # Аргумент перицентра
nu = 0.1 * np.pi  # Истиная аномалия
mu = 3.986 * 10 ** 14  # Гравитационный параметр
orbit = Orbit(a=a, e=e, i=i, omega=omega, w=w, nu=nu, mu=mu)

J = np.diag([2, 3, 4])  # Тензор инерции
Q0 = Quaternion(1, 0, 1, 0).normalised  # ориентация ССK относительно ИСК в нач момент времени
W0 = np.array([0.06, 0.07, 0.045])  # угловая относительная сокрость начальная в ССk
rotation = Rotation(J=J, Q0=Q0, W0=W0)

Kw = 1.4
Kq = 0.5
control = Control(Kw=Kw, Kq=Kq, J=J)

first_integrals = First_integrals(J=J, mu=mu)

# Моделируем
h = 0.01  # Шаг интегрирования по времени
n = 15000  # Количество шагов
simulation = Simulation(orbit=orbit, rotation=rotation, control=control, first_integrals=first_integrals)
simulation.start(h=h, n=n)

# строим графики
start_plot(orbit=orbit, rotation=rotation, first_integrals=first_integrals)
