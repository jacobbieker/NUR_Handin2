import numpy as np
import matplotlib.pyplot as plt


def integration_alg(func, lower_bound, upper_bound, number_of_steps):
    """

    :param func: function to use in integration, when given a radius, will return the value at that point
    :param lower_bound: lower bound of integration
    :param upper_bound: upper bound of integration
    :param number_of_steps: number of steps to do
    :return:
    """

    # Current method if the midpoint rule, as the function is an improper integral from the lower bound being 0

    # Need to integrate from lower_bound to upper_bound in number_of_steps

    # lower bound is 0, if there is a radius of 0, then no satallites in it
    integration_value = 0
    step_size = (upper_bound - lower_bound) / number_of_steps  # The number of steps to take
    for i in range(number_of_steps):
        if i != 0:
            # Current step can be just i*step_size but only if integral always starts at 0
            # since it might not, need a starting point otherwise:
            current_step = lower_bound + i * step_size
            prev_step = lower_bound + (i - 1) * step_size

            # Current midpoint is the current step + the prev step divided by 2
            # F(mk) where mk = (tk + tk-1)/2
            current_midpoint = (current_step + prev_step) / 2
            integration_value += func(current_midpoint)

    # Last bit is the multiplication by the step size to get the full value
    integration_value *= step_size

    return integration_value


def derivative(func, b, step_size=0.1, iterations=5):
    """
    This uses the central differences method to calculate the derivative of a function

    Ridder method: keep decreasing step_size until error grows

    A(1,m) = f(x+h/2^m-1) - f(x-h/2^m-1)/(2h/s^m-1)
    A(n,m) = 4^n-1*A(n-1,m+1) - A(n-1,m)/(4^n-1)-1)
    """

    def A_deriv(n, m):
        if n == 1:
            result = (func(b + step_size / 2 ** (m - 1)) - func(b - step_size / (2 ** (m - 1)))) / (
                    2 * step_size / (2 ** (m - 1)))
        else:
            result = (4 ** (n - 1) * A_deriv(n - 1, m + 1) - A_deriv(n - 1, m)) / (4 ** (n - 1) - 1)
        return result

    best_approx = np.inf
    m = 1
    for i in range(iterations):
        deriv = A_deriv(i + 1, m)
        m += 1
        if analytic_derivative(b) - deriv < best_approx:
            best_approx = deriv

    return deriv


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


def part_three(rand_gen):
    """
    Linear structure growth
    :param rand_gen:
    :return:
    """

    H0 = 7.16e-11
    Omega_0 = 1  # De-sitter Universe
    init_time = 1
    final_time = 1000
    time_step = 10
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

    # TODO Need to add analytic version
