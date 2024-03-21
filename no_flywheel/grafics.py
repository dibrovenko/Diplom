import numpy as np
from matplotlib import pyplot as plt
from src.orbit import Orbit
from src.rotation import Rotation
from src.first_integrals import First_integrals

# Функция для построения графика
def plot_subplot(subplot_num, data_x, data_y, title, x_label, y_label):
    plt.subplot(subplot_num)
    plt.plot(data_x, data_y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def save_and_close_plot(figure, filename):
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(figure)


def orbit_plot(orbit: Orbit):
    # Создаем фигуру и 3D-подграфик
    r_apofocus = 1.05 * orbit.result_a[0] * (1 + orbit.result_e[0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Создаем сферу, представляющую Землю
    r = 6371 * 1000  # Радиус Земли примерно 6371 км
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Отображаем сферу
    earth = ax.plot_surface(x, y, z, color='b', alpha=0.3, label='Земля')

    # Добавляем нулевой меридиан
    zero_meridian = np.array([r * np.cos(u), np.zeros(100), r * np.sin(u)])
    ax.plot(zero_meridian[0], zero_meridian[1], zero_meridian[2], color='b', linewidth=1, label='Нулевой меридиан')

    # Добавляем плоскость экватора
    xx, yy = np.meshgrid(np.linspace(-r_apofocus, r_apofocus, 10), np.linspace(-r_apofocus, r_apofocus, 10))
    zz = np.zeros(xx.shape)
    plane = ax.plot_surface(xx, yy, zz, color='g', alpha=0.5, label='Плоскость экватора')

    # Добавляем орбиту
    ax.plot(orbit.result_x, orbit.result_y, orbit.result_z, color='r', linewidth=3, label='Орбита спутника')

    # Добавляем точку старта
    ax.scatter(orbit.result_x[0], orbit.result_y[0], orbit.result_z[0], color='k', s=10, label='Точка старта')

    # Метки осей
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Масштабируйте оси
    ax.set_xlim(-r_apofocus, r_apofocus)
    ax.set_ylim(-r_apofocus, r_apofocus)
    ax.set_zlim(-r_apofocus, r_apofocus)

    # Отобразите график
    ax.legend()
    plt.tight_layout()
    plt.show()


def start_plot(orbit: Orbit, rotation: Rotation, first_integrals: First_integrals):
    # Графики первого интеграла С
    fig_c = plt.figure(figsize=(10.5, 7))
    plot_subplot(221, orbit.result_t, first_integrals.result_c[:, 0], 'Проекция первого интеграла С на Оx', 't, c', 'x')
    plot_subplot(222, orbit.result_t, first_integrals.result_c[:, 1], 'Проекция первого интеграла С на Оy', 't, c', 'y')
    plot_subplot(223, orbit.result_t, first_integrals.result_c[:, 2], 'Проекция первого интеграла С на Оz', 't, c', 'z')
    save_and_close_plot(fig_c, 'result/result_c.png')

    # Графики первого интеграла Лапласа
    fig_f = plt.figure(figsize=(10.5, 7))
    plot_subplot(221, orbit.result_t, first_integrals.result_f[:, 0], 'Проекция первого интеграла Лапласа на Оx', 't, c', 'x')
    plot_subplot(222, orbit.result_t, first_integrals.result_f[:, 1], 'Проекция первого интеграла Лапласа на Оy', 't, c', 'y')
    plot_subplot(223, orbit.result_t, first_integrals.result_f[:, 2], 'Проекция первого интеграла Лапласа на Оz', 't, c', 'z')
    save_and_close_plot(fig_f, 'result/result_f.png')

    # График интеграла энергии
    fig_E = plt.figure(figsize=(10.5, 7))
    plot_subplot(111, orbit.result_t, first_integrals.result_E, 'Интеграл энергии', 't, c', 'E')
    save_and_close_plot(fig_E, 'result/result_E.png')

    # График интеграла Якоби
    fig_E = plt.figure(figsize=(10.5, 7))
    plot_subplot(111, orbit.result_t, first_integrals.result_E, 'Интеграл Якоби', 't, c', 'h')
    save_and_close_plot(fig_E, 'result/result_h.png')

    # Графики орбитальных элементов
    fig_elements = plt.figure(figsize=(10.5, 7))
    plot_subplot(321, orbit.result_t, orbit.result_a, 'Большая полуось', 't, c', 'a')
    plot_subplot(322, orbit.result_t, orbit.result_e, 'Эксцентриситет', 't, c', 'e')
    plot_subplot(323, orbit.result_t, orbit.result_i, 'Наклонение', 't, c', 'i')
    plot_subplot(324, orbit.result_t, orbit.result_omega, 'Долгота восходящего узла ', 't, c', 'Ω')
    plot_subplot(325, orbit.result_t, orbit.result_w, 'Аргумент перицентра', 't, c', 'w')
    plot_subplot(326, orbit.result_t, orbit.result_nu, 'Истинная аномалия', 't, c', ' ν')
    save_and_close_plot(fig_elements, 'result/result_elements.png')

    # Cтроим орбиту
    orbit_plot(orbit=orbit)


