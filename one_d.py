import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm
from astropy.stats import kuiper as kuiper_reference

def common_test(points, func):
    """
    Calcs KS test with Scipy's norm
    :param points:
    :return:
    """
    number_of_bins = int(200*(max(points) - min(points)))
    values, bins = np.histogram(points, bins=number_of_bins)
    bin_width = bins[1] - bins[0]
    bins += bin_width

    return func(points, values, bins)

def box_muller(rand_gen, num_samples):
    z1 = []
    z2 = []
    for _ in range(num_samples):
        z1.append(next(rand_gen))
        z2.append(next(rand_gen))

    z1 = np.asarray(z1)
    z2 = np.asarray(z2)

    x1 = np.cos(2 * np.pi * z2) * np.sqrt((-2) * np.log(z1))
    return x1


def map_to_guass(x, u, sigma):
    # First change variance

    x = x * sigma

    # then change the mean
    x = x + u

    return x

def one_d(rand_gen):
    """
    Do Kuiper's test on the function

    This is z = F(xi)
    D+ = max(i/n-zi)
    D- = max(zi-(i-1)/n)
    V = D+ + D-

    :param rand_gen:
    :return:
    """

    def kuiper(z):
        if z < 0.4:
            return 1
        else:
            block = np.exp(-2*(z**2))
            return 2 *((4*(z**2-1))*block + (16*z**2-1)*block**4 + (32*z**2-1)*block**9)

    def kuiper_test_part(points, values, bins):
        summed_bins = sum(values)
        distribution = []
        for i in range(len(values)):
            distribution.append(abs(sum(values[:i])/summed_bins - norm.cdf(bins[i])))

        distribution = np.asarray(distribution)

        V = abs(max(distribution)) + abs(min(distribution))
        z = V*(np.sqrt(len(points)) + 0.155 + 0.24/np.sqrt(len(points)))

        return V, kuiper(z)

    sigma = 1
    u = 0
    num_samples = np.logspace(np.log10(10), np.log10(10**5), num=50)
    reference_ks = np.zeros(50)
    reference_p_value = np.zeros(50)
    test = np.zeros(50)
    p_value = np.zeros(50)

    for index, sample in enumerate(num_samples):
        sample = int(sample)
        gauss = box_muller(rand_gen, sample)
        gauss = map_to_guass(gauss, u=u, sigma=sigma)
        test[index], p_value[index] = common_test(gauss, kuiper_test_part)
        reference_ks[index], reference_p_value[index] = kuiper_reference(gauss, norm.cdf)

    plt.plot(num_samples, test, c='b', label='My Kuiper Test')
    plt.plot(num_samples, reference_ks, c='r', label='Astropy Kuiper Test')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Points")
    plt.ylabel("Kuiper Statistic (V)")
    plt.legend(loc='best')
    plt.savefig("plots/KuiperTest.png", dpi=300)
    plt.cla()

    plt.plot(num_samples, p_value, c='b', label='My Kuiper Test Probability')
    plt.plot(num_samples, reference_p_value, c='r', label='Astropy Kuiper Test Probability')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Points")
    plt.ylabel("Probability / 1 - p_value")
    plt.legend(loc='best')
    plt.savefig("plots/Kuiper_pvalue.png", dpi=300)
    plt.cla()
