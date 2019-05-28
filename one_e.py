import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def common_test(points, my_points, func):
    """
    Common test part for KS and Kuiper Test
    :param points:
    :return:
    """
    number_of_bins = 100
    values, bins = np.histogram(points, bins=number_of_bins)
    my_values = my_points

    return func(points, values, bins,  my_values)


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
    def bisect(arr, value):
        """
        Finds the index in the array closest to value
        :param arr: The array of values
        :param value: The value to insert/the interpolation value
        :return: Index of insertion point for the value in a sorted array
        """

        low = 0
        high = len(arr)
        while low < high:
            mid = int((low + high) / 2)  # Get the midpoint to test if the value is above or below it
            if value < arr[mid]:
                high = mid
            else:
                low = mid + 1
        return low

    def cdf1(value, points):
        """
        Computes CDF of points up to value
        :param values: Value to get the Cumsum at
        :param points:
        :return:
        """
        points = sorted(points) # Sort the points, so they are increasing
        # Could have used sorting method from last handin, but out of time
        cum_fun = np.asarray(list(range(len(points))))/len(points)

        # Now the index into it is the cumulative function up until now
        index = bisect(cum_fun, value)
        if index == len(cum_fun):
            return cum_fun[-1] # Returns last one
        else:
            return cum_fun[index]

    def cdf(value, points):
        """
        CDF Value, a lot faster than cdf1 because no sorting or doing binary search
        Takes advantage of numpy mask, cdf1 upped the runtime to nearly 10 minutes
        so this was chosen as an alternative

        :param value:
        :param points:
        :return:
        """
        return len(points[points <= value])/len(points)

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

    def ks_test_part(points, values, bins, my_values):
        summed_bins = sum(values)
        distribution = []
        for i in range(len(values)):
            distribution.append(abs(sum(values[:i]) / summed_bins - cdf(bins[i], my_values)))
            #distribution.append(abs(sum(values[:i]) / summed_bins - norm.cdf(bins[i])))

        distribution = np.asarray(distribution)

        D = max(distribution)
        z = D * (np.sqrt(len(points)) + 0.12 + 0.11 / np.sqrt(len(points)))

        return D, ks_test(z)

    def sciks_test_part(points, values, bins, my_values):
        summed_bins = sum(values)
        distribution = []
        for i in range(len(values)):
            #distribution.append(abs(sum(values[:i]) / summed_bins - cdf(bins[i], my_values)))
            distribution.append(abs(sum(values[:i]) / summed_bins - norm.cdf(bins[i])))

        distribution = np.asarray(distribution)

        D = max(distribution)
        z = D * (np.sqrt(len(points)) + 0.12 + 0.11 / np.sqrt(len(points)))

        return D, ks_test(z)

    numbers = np.loadtxt("randomnumbers.txt")
    num_sizes = len(numbers[:, 0])
    num_nums = len(numbers[0, :])

    samples = np.logspace(np.log10(10), np.log10(num_sizes), num=20).astype(np.int64)

    ks_tests = np.zeros(20)
    p_valus = np.zeros(20)
    sci_ks = np.zeros(20)
    scp_values = np.zeros(20)

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15, 15))
    fig1, (ax3, ax4) = plt.subplots(2,1, figsize=(15, 15))

    for i in range(num_nums):
        for j, size in enumerate(samples):
            gauss = box_muller(rand_gen, size)
            gauss = map_to_guass(gauss, u=0, sigma=1)
            # Could have used sorting method from last handin, but out of time
            ks_tests[j], p_valus[j] = common_test(numbers[:size, i], gauss, ks_test_part)
            sci_ks[j], scp_values[j] = common_test(numbers[:size, i], gauss, sciks_test_part)
        ax1.plot(samples, ks_tests, label='Set {}'.format(i))
        ax2.plot(samples, p_valus,  label='Set {}'.format((i)))
        ax3.plot(samples, sci_ks, label='Set {}'.format(i))
        ax4.plot(samples, scp_values,  label='Set {}'.format((i)))

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

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylim(1e-3,1)
    ax3.set_xlabel("Number of Points")
    ax3.set_ylabel("KS Statistic (D)")
    ax3.set_title("KS Test")
    ax3.legend(loc='best')
    ax4.set_xscale('log')
    ax4.set_ylim(1e-4, 1.2)
    #ax2.set_yscale('log')
    ax4.set_xlabel("Number of Points")
    ax4.set_ylabel("Probability")
    ax4.set_title("KS P values")
    ax4.legend(loc='upper left')
    fig1.suptitle("KS Test on Random Sets")

    fig1.savefig("plots/RandNumKS_sci.png", dpi=300)

    plt.cla()
    plt.close()


