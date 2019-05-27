import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def common_test(points, func):
    """
    Common test
    :param points:
    :return:
    """
    number_of_bins = int(200 * (max(points) - min(points)))
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

    all_ks = []
    all_p = []
    ks_tests = np.zeros(20)
    p_valus = np.zeros(20)

    for i in range(num_nums):
        for j, size in enumerate(samples):
            gauss = box_muller(rand_gen, size)
            gauss = map_to_guass(gauss, u=0, sigma=1)
            ks_tests[j], p_valus[j] = common_test(gauss, ks_test_part)
        all_ks.append(ks_tests)
        all_p.append(p_valus)

    for i in range(len(all_ks)):
        plt.plot(samples, all_ks[i], label='KS Test Set {}'.format(i))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Points")
    plt.ylabel("KS Statistic (D)")
    plt.title("KS Test For Rand Sets")
    plt.legend(loc='best')
    plt.savefig("plots/RandNumKS.png", dpi=300)
    plt.cla()

    for i in range(len(all_p)):
        plt.plot(samples, p_valus, label='KS P-Value Set {}'.format((i)))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Points")
    plt.ylabel("Probability / 1 - p_value")
    plt.title("KS Test For Rand Sets")
    plt.legend(loc='best')
    plt.savefig("plots/KStest_pvalue.png", dpi=300)
    plt.cla()
