import numpy as np
import matplotlib.pyplot as plt


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


def part_four_a(rand_gen):

    H0 = 7.16e-11
    Omega_lambda = 0.7
    Omega_m = 0.3

    def growth_factor_z(z):
        """
        Interior of integral
        :param z:
        :return:
        """
        return (1+z)/(Omega_m*(1+z)**3+Omega_Lambda)**(3/2)

    def growth_factor_a(a):
        """
        Growth Factor with a
        :param a:
        :return:
        """
        return (1/a**3)/(Omega_M/a**3 + Omega_Lambda)**(3/2)

    def H(z):
        """
        Hubble Constant
        :param z:
        :return:
        """
        return H0*np.sqrt(Omega_M*(1+z)**3 + Omega_Lambda)

    def D_a(a, A):
        """
        Linear Growth Factor D(a)
        :param a:
        :param A:
        :return:
        """
        return 5*Omega_M/2*(Omega_M/a**3 + Omega_Lambda)**(1/2)*A

    def differentiate_point(func, b, eps=1e-12):
        """
        Numerical differentiation at point
        :param func:
        :param b:
        :param eps:
        :return:
        """
        h = 0.01
        dydx = (func(b+h/2) - func(b - h/2)) / h

        while True:
            h = h / 2
            d = (func(b+h/2) - func(b - h/2)) / h
            if abs(d - dydx) < eps:
                return d
            else:
                dydx = d


    def linear_growth_factor(z):
        return (5*Omega_m*H0**2)/2*((H0**2)*(Omega_m*((1+z)**3) + Omega_lambda))**0.5

    def operand(z_prime):
        return (1/z_prime**2)*(1+1/z_prime)/((H0**2)*(Omega_m*(1+z_prime)**3 + Omega_lambda))**1.5


    a0 = 0
    a_final = 1/51

    A = integration_alg(growth_factor_a, a0, a_final)
    print("Integral value: {:.7e}".format(A))

    D = 5*Omega_M/2*np.sqrt(Omega_M/a_final**3 + Omega_Lambda)*A
    print("Linear growth factor at z = 50 (a = 1/51): {:.8e}".format(D))

def part_b():
    H0 = 7.16e-11
    Omega_lambda = 0.7
    Omega_m = 0.3

    def growth_factor_z(z):
        """
        Interior of integral
        :param z:
        :return:
        """
        return (1+z)/(Omega_m*(1+z)**3+Omega_Lambda)**(3/2)

    def growth_factor_a(a):
        """
        Growth Factor with a
        :param a:
        :return:
        """
        return (1/a**3)/(Omega_M/a**3 + Omega_Lambda)**(3/2)

    def H(z):
        """
        Hubble Constant
        :param z:
        :return:
        """
        return H0*np.sqrt(Omega_M*(1+z)**3 + Omega_Lambda)

    def D_a(a, A):
        """
        Linear Growth Factor D(a)
        :param a:
        :param A:
        :return:
        """
        return 5*Omega_M/2*(Omega_M/a**3 + Omega_Lambda)**(1/2)*A

    def differentiate_point(func, b, eps=1e-12):
        """
        Numerical differentiation at point
        :param func:
        :param b:
        :param eps:
        :return:
        """
        h = 0.01
        dydx = (func(b+h/2) - func(b - h/2)) / h

        while True:
            h = h / 2
            d = (func(b+h/2) - func(b - h/2)) / h
            if abs(d - dydx) < eps:
                return d
            else:
                dydx = d


    def linear_growth_factor(z):
        return (5*Omega_m*H0**2)/2*((H0**2)*(Omega_m*((1+z)**3) + Omega_lambda))**0.5

    def operand(z_prime):
        return (1/z_prime**2)*(1+1/z_prime)/((H0**2)*(Omega_m*(1+z_prime)**3 + Omega_lambda))**1.5


    a0 = 0
    a_final = 1/51
    A = integration_alg(growth_factor_a, a0, a_final)

    D_prime = -15/4*Omega_M**2*H0*A/a_final**3
    print('First derivative of LGF at z = 50 (a = 1/51): {:.8e}'.format(D_prime))

    D_prime_numerical = a_final*H(1/a_final - 1)*differentiate_point(D_a, b=a_final, eps=1e-13)
    print('First derivative of LGF at z = 50 (a = 1/51), numerical: {:.8e}'.format(D_prime_numerical))
