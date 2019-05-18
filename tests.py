def ks_test(z):
    for i in range(len(z)):
        if z[i]<1.18:
            p_ks = (np.sqrt(2*np.pi)/z)* \
                   ((np.exp(-np.pi**2/(8*z**2)))+(np.exp(-np.pi**2/(8*z**2)))**9+(np.exp(-np.pi**2/(8*z**2)))**25)
        else:
            p_ks = 1-2*((np.exp(-2*z**2))-(np.exp(-2*z**2))**4+(np.exp(-2*z**2))**9)

    return p_ks

from scipy.stats import kstest

print(kstest(normal, 'norm'))

#print(ks_test(normal))

# Give the parameters.
H0 = 7.16e-11 # [yr^-1]
omega_0 = 1

# Define a and time-derivative of a.
a = lambda t: ((3/2)*H0*t)**(2/3)
a_dot = lambda t: H0*((3/2)*H0*t)**(-1/3)

# Define the function to be solved.
def ode(r, t):

    """
    r ... A variable containing D and dD/dt
    t ... time to integrate over

    """

    # Define the initial conditions. Will assign values later.
    D = r[0]
    dD_dt = r[1]

    d2D_dt2 = -2*(a_dot(t)/a(t))*dD_dt + (3/2)*omega_0*(H0**2)*(D/(a(t)**3))

    return np.array([dD_dt, d2D_dt2])

# define the solver method.
def rk4(ode,r,t,h):

    """
    ode ... ordinary differential equation to solve
    r   ... A variable containing D and dD/dt
    t   ... time to integrate over
    h   ... integration step

    Method using here is the classic Runge-Kutta (4-th order), acquired L10.

    """

    k1 = h * ode(r, t)
    k2 = h * ode(r+k1/2., t+h/2.)
    k3 = h * ode(r+k2/2., t+h/2.)
    k4 = h * ode(r+k3, t+h)

    return k1/6. + k2/3. + k3/3. + k4/6.

# Define the ode solver.
def ode_solver(init, lower, upper, N):

    # Define the integration step.
    h = (upper-lower)/N

    # Define initial conditions and assign values.
    r = init
    t_values, D_values = [],[]

    # Solve the ode using rk4.
    for t in np.arange(lower, upper, h):
        t_values.append(t)
        D_values.append(r[0])
        r += rk4(ode,r,t,h)

    return t_values, D_values


# Define the time range and number of data points.
t_init, t_final, t_step = 1., 1000., 10
N = (t_final-t_init)*t_step

# Plot given the initial conditions.
plt.figure()

case1, case2, case3 = [3,2], [10,-10], [5,0]
all_cases = [[case1,'case1'], [case2,'case2'], [case3,'case3']]

for i in range(len(all_cases)):
    t_values, D_values = ode_solver(all_cases[i][0], t_init, t_final, N)
    plt.loglog(t_values, D_values, label=all_cases[i][1])

plt.xlabel('t', fontsize=14)
plt.ylabel('D(t)', fontsize=14)
plt.legend()

H0 = 7.16e-11 # [yr^-1]
omega_lambda = 0.7
omega_m = 0.3

#H_z = lambda z: (H0**2)*(omega_m*(1+z)**3 + omega_lambda)
linear_growth_factor = lambda z: (5*omega_m*H0**2/2)*((H0**2)*(omega_m*(1+z)**3 + omega_lambda))**(1/2)

operand = lambda z_prime: (1/z_prime**2)*((1+1/z_prime)/(((H0**2)*(omega_m*(1+z_prime)**3 + omega_lambda))**(3/2)))

# Midpoint rule.
def midpoint(f, a, b, N):
    """
    f     ... function to be integrated
    [a,b] ... integration interval
    N     ... number steps(bigger the better but slower)

    """

    h = float(b-a)/N
    output = 0
    for i in range(N):
        output += f((a+h/2.0)+i*h)
    output *= h
    return output

