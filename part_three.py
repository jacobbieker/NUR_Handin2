import numpy as np
import matplotlib.pyplot as plt

H0 = 7.16e-11
Omega_0 = 1


def a(t):
    return ((3 / 2) * H0 * t) ** (2 / 3)


def a_dot(t):
    return H0 * ((3 / 2) * H0 * t) ** (-1 / 3)


def dadt(t):
    return ((2*H0**2)/(3*t))**(1/3)


def second_order(r, t):
    D = r[0]
    y = r[1]
    dD = y
    dy = (2/3)*D/t**2 - 4*y/(3*t)
    return np.array([dD, dy])


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

    return r + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6


def D_analytic_solution(t, case):
    """
    Analytic Solutions for D based on the three cases
    :param t:
    :param case:
    :return:
    """
    if case == "case 1":
        return 3*t**(2/3)
    elif case == "case 2":
        return 10/t
    elif case == "case 3":
        return 3*t**(3/2) + 2/t
    else:
        raise NotImplementedError


def part_three():
    """
    Linear structure growth
    :param rand_gen:
    :return:
    """
    t = np.linspace(1, 1000, 10000)
    D = np.zeros(len(t))
    y = np.zeros(len(t))    # y = dD/dt

    case_one = [3, 2]
    case_two = [10, -10]
    case_three = [5, 0]

    cases = ([case_one, "case 1"], [case_two, "case 2"], [case_three, "case 3"])
    for i in range(len(cases)):
        #times, Ds = solver(cases[i][0], times)
        D[0] = cases[i][0][0]
        y[0] = cases[i][0][1]

        h = t[1] - t[0]

        for j in range(1, len(t)):
            r = np.array([D[j-1], y[j-1]])
            r = runge_kutta(second_order, r, t[j-1], h)
            D[j] = r[0]
            y[j] = r[1]

        plt.plot(t, D, label=cases[i][1])
        plt.plot(t, D_analytic_solution(t, cases[i][1]), linestyle="--", label="Analytic {}".format(cases[i][1]))

    plt.title("Numerical and Analytic Solutions")
    plt.xlabel("Time (yr)")
    plt.ylabel("Linear Growth Factor D(t)")
    plt.legend(loc='best')
    plt.loglog()
    plt.savefig("plots/growth_factors.png", dpi=300)
    plt.cla()

part_three()