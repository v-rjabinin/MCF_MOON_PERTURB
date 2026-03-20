from classes import Orbit, CoordTransformer, MoonAccelerComputer, Float, Int
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import numpy as np
import data


def calculate_eccent_anom(M: Float, e: Float, eps: Float = 1e-6, iter_lim: Int = 1000) -> Float:
    curr_anom, i = M, 0
    next_anom = M + e * np.sin(curr_anom)

    while np.any(np.abs(next_anom - curr_anom) > eps) and (i < iter_lim):
        curr_anom = next_anom
        next_anom = M + e * np.sin(curr_anom)
        i += 1

    return next_anom


def calculate_eccent_anom_from_true_anom(nu: Float, e: Float) -> Float:
    E =  2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))

    if E < 0:
        E += 2 * np.pi

    return E


def calculate_true_anom_from_eccent_anom(E: Float, e: Float) -> Float:
    nu =  2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

    if nu < 0:
        nu += 2 * np.pi

    return nu


def set_pi_axis(ax, step=np.pi / 2):
    def pi_formatter(x, pos):
        if x == 0:
            return '0'

        coeff = x / np.pi
        coeff_rounded = np.round(coeff, 3)

        standard_fractions = {
            0.25: (r'$\frac{\pi}{4}$', 1, 4),
            0.5: (r'$\frac{\pi}{2}$', 1, 2),
            0.75: (r'$\frac{3\pi}{4}$', 3, 4),
            1.0: (r'$\pi$', 1, 1),
            1.25: (r'$\frac{5\pi}{4}$', 5, 4),
            1.5: (r'$\frac{3\pi}{2}$', 3, 2),
            1.75: (r'$\frac{7\pi}{4}$', 7, 4),
            2.0: (r'$2\pi$', 2, 1),
        }

        for std_coeff, (latex_str, n, m) in standard_fractions.items():
            if np.abs(coeff_rounded - std_coeff) < 0.001:
                return latex_str

        if np.abs(coeff_rounded - int(coeff_rounded)) < 0.01:
            n = int(np.round(coeff_rounded))
            if n == 1:
                return r'$\pi$'
            elif n == -1:
                return r'$-\pi$'
            else:
                return rf'${n}\pi$'

        for m in [2, 3, 4, 6, 8]:
            n_float = coeff_rounded * m
            n_rounded = np.round(n_float)
            if np.abs(n_float - n_rounded) < 0.01 and n_rounded != 0:
                n = int(n_rounded)

                gcd = np.gcd(n, m)
                n_simplified = n // gcd
                m_simplified = m // gcd

                if m_simplified == 1:
                    if n_simplified == 1:
                        return r'$\pi$'
                    elif n_simplified == -1:
                        return r'$-\pi$'
                    else:
                        return rf'${n_simplified}\pi$'

                if n_simplified == 1 and m_simplified == 2:
                    return r'$\frac{\pi}{2}$'
                elif n_simplified == -1 and m_simplified == 2:
                    return r'$-\frac{\pi}{2}$'
                elif m_simplified == 2:
                    return rf'$\frac{{{n_simplified}\pi}}{{2}}$'
                elif m_simplified == 4:
                    return rf'$\frac{{{n_simplified}\pi}}{{4}}$'
                else:
                    return rf'$\frac{{{n_simplified}\pi}}{{{m_simplified}}}$'

        return rf'${coeff_rounded:.2f}\pi$'

    ax.xaxis.set_major_formatter(plt.FuncFormatter(pi_formatter))
    ax.xaxis.set_major_locator(plt.MultipleLocator(base=step))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')


def system(u, y):
    # Parameters to integrate:
    # p:            Focal parameter, (km)
    # e:            Eccentricity, (no dim)
    # i:            Inclination, (rad)
    # omega:        Longitude of the ascending node, (rad)
    # arg_per:      Argument of periapsis, (rad)
    # t:            Time, (s)

    p, e, i, omega, arg_per, t = y
    a, nu = p / (1 - e * e), u - arg_per
    r = p / (1 + e * np.cos(nu))

    pi_2, mu = 2 * np.pi, data.mu

    M_moon = data.n_moon * (t - data.tao_moon)
    E_moon = calculate_eccent_anom(M_moon, data.e_moon)
    u_moon = calculate_true_anom_from_eccent_anom(E_moon, data.e_moon) + data.arg_per_moon

    sat_orbit = Orbit(a, e, i, np.array([u]) % pi_2, omega, arg_per % pi_2)
    moon_orbit = Orbit(data.a_moon, data.e_moon, data.i_moon, np.array([u_moon]) % pi_2, data.omega_moon, data.arg_per_moon)

    sat_coord, moon_coord = CoordTransformer(sat_orbit), CoordTransformer(moon_orbit)

    sat_coord.transform_to_gocs()

    moon_coord.transform_to_eci()
    moon_coord.transform_to_external_gocs(omega, i, arg_per % pi_2)

    acceler_computer = MoonAccelerComputer(sat_coord, moon_coord, sat_orbit)
    acceler, distance = acceler_computer()

    curr_point.append(u)
    acceleration.append(acceler)
    dist.append(distance)

    ctg = lambda x: np.cos(x) / np.sin(x)

    do_du = (r ** 3) * np.sin(u) * acceler[2] / (mu * p * np.sin(i))
    di_du = (r ** 3) * np.cos(u) * acceler[2] / (mu * p)
    dp_du = 2 * (r ** 3) * acceler[1] / mu
    de_du = (r ** 2) * (np.sin(nu) * acceler[0] + np.cos(nu) * (1 + r / p) * acceler[1] + e * acceler[2] * r / p) / (mu * e)
    dw_du = (r ** 2) * (np.cos(nu) * acceler[0] + e * np.sin(nu) * (1 + r / p) * acceler[1] - e * ctg(i) * np.sin(u) * acceler[2] * r / p) / (mu * e)

    dnu_dt = np.sqrt(mu * p) / (r ** 2)
    dt_du = 1 / dnu_dt

    return [dp_du[0], de_du[0], di_du[0], do_du[0], dw_du[0], dt_du]


if __name__ == "__main__":
    acceleration, curr_point, dist = list(), list(), list()

    u_segment = [data.u, data.u + 10 * np.pi]
    initial_data = [data.p, data.e, data.i, data.omega, data.arg_per, data.t_start]

    solution = solve_ivp(system, u_segment, initial_data, dense_output=True, rtol=1e-12, atol=1e-12)

    u_values = solution.t

    p_values = solution.y[0, :]
    e_values = solution.y[1, :]
    i_values = solution.y[2, :]
    omega_values = solution.y[3, :]
    arg_per_values = solution.y[4, :]
    t_values = solution.y[5, :]

    values_1, values_2 = [p_values, e_values, i_values, omega_values, arg_per_values], []
    titles_1 = ["Фокальный параметр (p), км", "Эксцентриситет (e), -", "Наклонение орбиты (i), рад", "Долгота восходящего узла (Ω), рад", "Аргумент широты перицентра (ω), рад", r"Расстояние между Луной и КА |$\vec{r}_2$|, км"]
    titles_2 = ["Время (t), сек", r"Радиальная компонента $\vec{a}_{возм}$, $км/c^2$", r"Трансверсальная компонента $\vec{a}_{возм}$, $км/c^2$", r"Бинормальная компонента $\vec{a}_{возм}$, $км/c^2$", r"Возмущающее ускорение $\vec{a}_{полн}$, $км/c^2$", r"Расстояние между Луной и КА |$\vec{r}_2$|, км"]

    acceleration, curr_point, dist = np.hstack(acceleration), np.array(curr_point, dtype=np.float64), np.array(dist, dtype=np.float64)

    value_to_index = {val: idx for idx, val in enumerate(curr_point)}
    indices = np.array([value_to_index[val] for val in u_values])

    acceleration, dist = acceleration[:, indices], dist[indices]

    values_1.append(dist)
    values_2.extend([t_values, acceleration[0, :], acceleration[1, :], acceleration[2, :], np.sqrt(np.sum(acceleration ** 2, 0)), dist])

    fig, ax = plt.subplots(2, 3, figsize=(20, 10))

    for i, axis in enumerate(ax.ravel()):
        axis.plot(u_values, values_1[i], "-r")

        axis.grid(True)

        axis.set_title(titles_1[i], fontsize=11, fontstyle='italic')
        axis.set_xlabel("u, рад", fontsize=10)

        set_pi_axis(axis, np.pi)
        axis.yaxis.set_major_locator(plt.MaxNLocator(8))

    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.tight_layout()

    fig, ax = plt.subplots(2, 3, figsize=(20, 10))

    for i, axis in enumerate(ax.ravel()):
        axis.plot(u_values, values_2[i], "-b")

        axis.grid(True)

        axis.set_title(titles_2[i], fontsize=11, fontstyle='italic')
        axis.set_xlabel("u, рад", fontsize=10)

        set_pi_axis(axis, np.pi)
        axis.yaxis.set_major_locator(plt.MaxNLocator(8))

    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.tight_layout()

    plt.show()