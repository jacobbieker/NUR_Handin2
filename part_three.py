import numpy as np
import matplotlib.pyplot as plt

H0 = 7.16e-11
Omega_0 = 1


def a(t):
    return ((3 / 2) * H0 * t) ** (2 / 3)


def a_dot(t):
    return H0 * ((3 / 2) * H0 * t) ** (-1 / 3)


def diff_eqn(r, t):
    """

    :param r: variable containing D and dD/dt
    :param t: time to integrate over
    :return:
    """

    init_D = r[0]
    dD_dt = r[1]

    d2D_dt2 = -2 * (a_dot(t) / a(t)) * (dD_dt) + (3 / 2) * Omega_0 * H0 ** 2 * (1 / a(t) ** 3) * init_D

    return np.array([dD_dt, d2D_dt2])


def runge_kutta(diff_egn, r, t, h):
    """
    4th order Runge-Kutta method

    :param diff_egn:
    :param r:
    :param t:
    :param h:
    :return:
    """

    k1 = h * diff_egn(r, t)
    k2 = h * diff_egn(r + k1 / 2, t + h / 2)
    k3 = h * diff_egn(r + k2 / 2, t + h / 2)
    k4 = h * diff_egn(r + k3, t + h)

    return k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6


def solver(init, start, end, num_points):
    """
    Solves the ODE
    :param init:
    :param start:
    :param end:
    :param num_points:
    :return:
    """

    # integration step
    step = (end - start) / num_points

    # initial conditions
    r = init
    times = []
    Ds = []

    # Solve ODE
    for t in np.arange(start, end, step):
        times.append(t)
        Ds.append(r[0])
        r += runge_kutta(diff_eqn, r, t, step)

    return times, Ds


def part_three():
    """
    Linear structure growth
    :param rand_gen:
    :return:
    """

    H0 = 7.16e-11
    Omega_0 = 1  # De-sitter Universe
    init_time = 1
    final_time = 1000
    time_step = 1
    num_steps = int((final_time - init_time) / time_step)

    case_one = [3, 2]
    case_two = [10, -10]
    case_three = [5, 0]

    cases = ([case_one, "case 1"], [case_two, "case 2"], [case_three, "case 3"])

    for i in range(len(cases)):
        times, Ds = solver(cases[i][0], init_time, final_time, num_steps)
        plt.loglog(times, Ds, label=cases[i][1])

    plt.xlabel("Time (yr)")
    plt.ylabel("D(t)")
    plt.legend(loc='best')
    plt.show()

    # TODO Need to add analytic version and plot that one


part_three()