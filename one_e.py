import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def common_test(points, func):
    """
    Common test part for KS and Kuiper Test
    :param points:
    :return:
    """
    number_of_bins = int(100 * (max(points) - min(points)))
    values, bins = np.histogram(points, bins=number_of_bins)

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


def one_e(rand_gen):
    """
    Compare to downloaded random numbers
    :param rand_gen:
    :return:
    """

    def ks_test(z):
        if z == 0:
            return 1
        elif z < 1.18:  # Numerically optimal cutoff
            block = ((np.exp((-1. * np.pi ** 2) / (8 * z ** 2))))
            p_ks = (np.sqrt(2 * np.pi) / z) * \
                   (block + block ** 9 + block ** 25)
        else:
            block = np.exp(-2 * z ** 2)
            p_ks = 1 - 2 * (block - block ** 4 + block ** 9)
        return 1 - p_ks

    def ks_test_part(points, values, bins):
        summed_bins = sum(values)
        distribution = []
        for i in range(len(values)):
            distribution.append(abs(sum(values[:i]) / summed_bins - norm.cdf(bins[i])))

        distribution = np.asarray(distribution)

        D = max(abs(distribution))
        z = D * (np.sqrt(len(points)) + 0.12 + 0.11 / np.sqrt(len(points)))

        return D, ks_test(z)

    numbers = np.loadtxt("randomnumbers.txt")
    num_sizes = len(numbers[:, 0])
    num_nums = len(numbers[0, :])

    samples = np.logspace(np.log10(10), np.log10(num_sizes), num=20).astype(np.int64)

    ks_tests = np.zeros(20)
    p_valus = np.zeros(20)

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15, 15))
    #fig2, ax = plt.subplots(1,1)

    for i in range(num_nums):
        for j, size in enumerate(samples):
            gauss = box_muller(rand_gen, size)
            gauss = map_to_guass(gauss, u=0, sigma=1)
            ks_tests[j], p_valus[j] = common_test(gauss, ks_test_part)
        ax1.plot(samples, ks_tests, label='Set {}'.format(i))
        ax2.plot(samples, p_valus,  label='Set {}'.format((i)))

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-3,1)
    ax1.set_xlabel("Number of Points")
    ax1.set_ylabel("KS Statistic (D)")
    ax1.set_title("KS Test")
    ax1.legend(loc='best')
    ax2.set_xscale('log')
    ax2.set_ylim(1e-4, 1.2)
    #ax2.set_yscale('log')
    ax2.set_xlabel("Number of Points")
    ax2.set_ylabel("Probability")
    ax2.set_title("KS P values")
    ax2.legend(loc='upper left')
    fig.suptitle("KS Test on Random Sets")

    fig.savefig("plots/RandNumKS.png", dpi=300)

    plt.cla()
    plt.close()
