import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm


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


def common_test(points, func):
    """
    Common part for Kuiper and KS Test
    :param points:
    :return:
    """
    number_of_bins = 100
    values, bins = np.histogram(points, bins=number_of_bins)

    return func(points, values, bins)


def one_c(rand_gen):
    """
    KS Test
    :param rand_gen:
    :return:
    """

    # Now need to do the ks test
    # This calculates the value for KS at given points
    def ks_test(z):
        if z == 0:
            return 1
        elif z < 1.18:  # Numerically optimal cutoff
            block = ((np.exp((-1. * np.pi ** 2) / (8 * z ** 2))))
            p = (np.sqrt(2 * np.pi) / z) * \
                   (block + block ** 9 + block ** 25)
        else:
            block = np.exp(-2 * z ** 2)
            p = 1 - 2 * (block - block ** 4 + block ** 9)
        return 1 - p

    def ks_test_part(points, values, bins):
        summed_bins = sum(values)
        distribution = []
        for i in range(len(values)):
            distribution.append(abs(sum(values[:i]) / summed_bins - norm.cdf(bins[i])))

        distribution = np.asarray(distribution)

        D = max(distribution)
        z = D * (np.sqrt(len(points)) + 0.12 + 0.11 / np.sqrt(len(points)))

        return D, ks_test(z)

    sigma = 1
    u = 0
    num_samples = np.logspace(np.log10(10), np.log10(10 ** 5), num=50)
    reference_ks = np.zeros(50)
    reference_p_value = np.zeros(50)
    ks = np.zeros(50)
    p_value = np.zeros(50)

    for index, sample in enumerate(num_samples):
        sample = int(sample)
        gauss = box_muller(rand_gen, sample)
        gauss = map_to_guass(gauss, u=u, sigma=sigma)
        ks[index], p_value[index] = common_test(gauss, ks_test_part)
        reference_ks[index], reference_p_value[index] = kstest(gauss, "norm")

    plt.plot(num_samples, ks, c='b', label='My KS Test')
    plt.plot(num_samples, reference_ks, c='r', label='Scipy KS Test')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Points")
    plt.ylabel("KS Statistic (D)")
    plt.legend(loc='best')
    plt.savefig("plots/KStest.png", dpi=300)
    plt.cla()

    plt.plot(num_samples, p_value, c='b', label='My KS Test Probability')
    plt.plot(num_samples, reference_p_value, c='r', label='Scipy KS Test Probability')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Points")
    plt.ylabel("Probability")
    plt.legend(loc='best')
    plt.savefig("plots/KStest_pvalue.png", dpi=300)
    plt.cla()
