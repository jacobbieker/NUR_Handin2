import numpy as np
import sys

H0 = 7.16e-11
Omega_M = 0.3
Omega_Lambda = 0.7


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


def part_b():

    def a_from_z(z):
        """
        Gets a from z
        :param z:
        :return:
        """
        return 1/(z+1)

    def growth_factor_a(a):
        """
        Growth Factor with a
        :param a:
        :return:
        """
        return (1 / a ** 3) / (Omega_M / a ** 3 + Omega_Lambda) ** (3 / 2)

    def H(a):
        """
        Hubble Constant with respect to a
        :param a:
        :return:
        """
        return H0 * np.sqrt(Omega_M * (1/a) ** 3 + Omega_Lambda)

    def D_a(a, integral):
        """
        Linear Growth Factor D(a) with precalced integral
        :param a:
        :param integral:
        :return:
        """
        return 5 * Omega_M / 2 * (Omega_M / a ** 3 + Omega_Lambda) ** (1 / 2) * integral

    def differentiate_alg(func, b, integral, eps=1e-12):
        """
        Numerical differentiation at point by calculating deriviative at a point and halving step size

        :param func:
        :param b:
        :param eps:
        :return:
        """

        h = 0.01
        # First step
        def calc_deriv():
            return (func(b + h / 2, integral) - func(b - h / 2, integral)) / h
        prev_deriv = calc_deriv()

        while True:
            h = h / 2 # Cut step size in half each iteration
            deriv = calc_deriv()
            if abs(deriv - prev_deriv) < eps: # Means it has converged
                return deriv
            else:
                prev_deriv = deriv

    a0 = 0
    final_a = a_from_z(50) # z = 50, a = 1/(z+1) (z+1) = 1/a
    integral = integration_alg(growth_factor_a, a0, final_a, 20000)
    sys.stdout = open('4b.txt', 'w')
    numerical_deriv = final_a * H(final_a) * differentiate_alg(D_a, b=final_a, integral=integral)
    print('Numerical d/da of Linear Growth Factor at z = 50: {}'.format(numerical_deriv))
    analytic_deriv = -15 / 4 * Omega_M ** 2 * H0 * integral / final_a ** 3
    print('Analytical d/da of Linear Growth Factor at z = 50: {}'.format(analytic_deriv))
