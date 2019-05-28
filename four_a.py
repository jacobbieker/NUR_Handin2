import numpy as np
import sys


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


def part_four_a():
    Omega_M = 0.3
    Omega_Lambda = 0.7

    def integrand(a):
        """
        Integrand for the LGF
        :param a:
        :return:
        """
        return (1 / a ** 3) / (Omega_M / a ** 3 + Omega_Lambda) ** 1.5

    def a_from_z(z):
        """
        Gets a from z
        :param z:
        :return:
        """
        return 1 / (z + 1)

    a0 = 0
    final_a = a_from_z(50)  # z = 50, a = (z+1)
    sys.stdout = open('4a.txt', 'w')
    integral = integration_alg(integrand, a0, final_a, 20000)
    print("Integral value for the Integrand: {}".format(integral))

    lgf = 5 * Omega_M / 2 * np.sqrt(Omega_M / final_a ** 3 + Omega_Lambda) * integral
    print("Linear growth factor at z = 50: {}".format(lgf))
