import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm
from astropy.stats import kuiper as kuiper_reference

def random_generator(seed, m=2 ** 64 - 1, a=2349543, c=913842, a1=21, a2=35, a3=4, a4=4294957665):
    """
        Generates psuedorandom numbers with a combination of (M)LCC, 64 bit shift, and MWC
    :param seed: Seed to use
    :param m: Determines period of the MLCC
    :param a: For the MLCC
    :param c: For the MLCC
    :param a1: For the first bit shift
    :param a2: For the second bit shift
    :param a3: For the third bit shift
    :param a4: For the MWC
    :return:
    """

    # First linear congruential generator
    # While true, so the generator never stops making new numbers
    # This is used to make sure teh XOR shift is 64 bit
    bit_64 = 0xffffffffffffffff
    while True:
        # This is MLCC part
        generated_number = (a * seed + c) % m
        # Now bit shift
        generated_number = generated_number ^ (generated_number >> a1) & bit_64
        generated_number = generated_number ^ (generated_number << a2) & bit_64
        generated_number = generated_number ^ (generated_number >> a3) & bit_64

        # Now MWC part
        mwc_out = a4 * (generated_number & (2 ** 32 - 1)) + (generated_number >> 32)

        seed = mwc_out  # set the seed to a new number, so a different number generated next time
        mwc_out = mwc_out / m

        if mwc_out > 1.:
            # Have to make it between 1 and 0, so mod 1. makes sure its between 0 and 1 now
            close_to_final = mwc_out % 1.
        else:
            close_to_final = mwc_out

        yield close_to_final


def one_a(rand_gen):
    first_thousand = []
    first_thousand_x_1 = [0.0]
    # 1b first one
    for i in range(1000):
        # This is x_i+1
        first_thousand.append(next(rand_gen))
        if i > 0:
            # This is x_i
            first_thousand_x_1.append(first_thousand[i - 1])

    # Now plot xi+1 vs xi

    plt.scatter(first_thousand, first_thousand_x_1)
    plt.xlabel("$X_i$")
    plt.ylabel("$X_{i+1}$")
    plt.title("$X_{i+1}$ vs. $X_i$")
    plt.savefig("./plots/Xi_Xi_1.png", dpi=300)
    plt.cla()

    # Now plot xi vs the index of xi
    plt.scatter(np.arange(1000), first_thousand)
    plt.xlabel("Index")
    plt.ylabel("$X_i$")
    plt.title("Index vs. $X_i$")
    plt.savefig("./plots/Index_Xi_1.png", dpi=300)
    plt.cla()

    first_million = []
    for i in range(1000000):
        first_million.append(next(rand_gen))

    plt.hist(first_million, bins=np.linspace(0.0, 1.0, 21))
    plt.xlabel("Generated Number")
    plt.ylabel("Number of Elements")
    plt.savefig("./plots/1000000_rand.png", dpi=300)
    plt.cla()


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

def normpdf(x):
    top = np.exp(-x**2/2)
    bottom = np.sqrt(2*np.pi)
    return top/bottom

def one_b(rand_gen):
    """
    Box Muller Method
    :param rand_gen:
    :return:
    """
    sigma = 2.4
    u = 3
    gauss = box_muller(rand_gen, 1000)
    gauss = map_to_guass(gauss, u=u, sigma=sigma)

    def gaussian(x, u=0, variance=1.):
        return 1 / (np.sqrt(2 * np.pi * variance)) * np.exp(-(x - u) ** 2 / (2 * variance))

    pdf_x = np.linspace(u - 5*sigma, u + 5*sigma, 10000)
    pdf_y = gaussian(pdf_x, u, sigma*sigma)

    plt.plot(pdf_x, pdf_y, c='r', label='Gaussian PDF')
    plt.hist(gauss, density=True, bins=12, label='Box-Muller')
    plt.axvline(x=u + sigma, c='r', ls='dashed', label='1$\sigma$')
    plt.axvline(x=u - sigma, c='r', ls='dashed')
    plt.axvline(x=u - 2 * sigma, c='purple', ls='dashed', label='2$\sigma$')
    plt.axvline(x=u - 3 * sigma, c='g', ls='dashed', label='3$\sigma$')
    plt.axvline(x=u - 4 * sigma, c='y', ls='dashed', label='4$\sigma$')
    plt.axvline(x=u + 2 * sigma, c='purple', ls='dashed')
    plt.axvline(x=u + 3 * sigma, c='g', ls='dashed')
    plt.axvline(x=u + 4 * sigma, c='y', ls='dashed')
    plt.xlim(u - 5 * sigma, u + 5 * sigma)
    plt.legend(loc='best')
    plt.savefig("./plots/box_gauss.png", dpi=300)
    plt.cla()

one_b(random_generator(5227))


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

def one_c(rand_gen):
    """
    KS Test
    :param rand_gen:
    :return:
    """

    # Now need to do the ks test
    # This calculates the value for KS at given points
    def ks_test(z):
        #for i in range(len(z)):
        if z == 0:
            return 1
        elif z < 1.18: # Numerically optimal cutoff
            block = ((np.exp((-1.*np.pi**2) / (8 * z ** 2))))
            p_ks = (np.sqrt(2*np.pi) / z) * \
                   (block + block**9 + block**25)
        else:
            block = np.exp(-2 * z ** 2)
            p_ks = 1 - 2*(block - block**4 + block**9)
        return 1 - p_ks

    def ks_test_part(points, values, bins):
        summed_bins = sum(values)
        distribution = []
        for i in range(len(values)):
            distribution.append(abs(sum(values[:i])/summed_bins - norm.cdf(bins[i])))

        distribution = np.asarray(distribution)

        D = max(abs(distribution))
        z = D*(np.sqrt(len(points)) + 0.12 + 0.11/np.sqrt(len(points)))

        return D, ks_test(z)

    sigma = 1
    u = 0
    probs = []
    real_probs = []
    num_samples = np.logspace(np.log10(10), np.log10(10**5), num=50)
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
    plt.ylabel("Probability / 1 - p_value")
    plt.legend(loc='best')
    plt.savefig("plots/KStest_pvalue.png", dpi=300)
    plt.cla()

one_c(random_generator(5227))

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

one_d(random_generator(5227))

def one_e(rand_gen):
    """
    Compare to downloaded random numbers
    :param rand_gen:
    :return:
    """

    def ks_test(z):
        #for i in range(len(z)):
        if z == 0:
            return 1
        elif z < 1.18: # Numerically optimal cutoff
            block = ((np.exp((-1.*np.pi**2) / (8 * z ** 2))))
            p_ks = (np.sqrt(2*np.pi) / z) * \
                   (block + block**9 + block**25)
        else:
            block = np.exp(-2 * z ** 2)
            p_ks = 1 - 2*(block - block**4 + block**9)
        return 1 - p_ks

    def ks_test_part(points, values, bins):
        summed_bins = sum(values)
        distribution = []
        for i in range(len(values)):
            distribution.append(abs(sum(values[:i])/summed_bins - norm.cdf(bins[i])))

        distribution = np.asarray(distribution)

        D = max(abs(distribution))
        z = D*(np.sqrt(len(points)) + 0.12 + 0.11/np.sqrt(len(points)))

        return D, ks_test(z)

    numbers = np.loadtxt("randomnumbers.txt")
    num_sizes = len(numbers[:,0])
    num_nums = len(numbers[0,:])

    samples = np.logspace(np.log10(10), np.log10(num_sizes), num=20).astype(np.int64)

    ks_tests = np.zeros(20)
    p_valus = np.zeros(20)

    for i in range(num_nums):
        for j, size in enumerate(samples):
            gauss = box_muller(rand_gen, size)
            gauss = map_to_guass(gauss, u=0, sigma=1)
            ks_tests[j], p_valus[j] = common_test(gauss, ks_test_part)


        plt.plot(samples, ks_tests, c='b', label='KS Test Statistic')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Number of Points")
        plt.ylabel("KS Statistic (D)")
        plt.title("KS Test For Rand Set: {}".format(i))
        plt.legend(loc='best')
        plt.savefig("plots/RandNumKS_{}.png".format(i), dpi=300)
        plt.cla()

        plt.plot(samples, p_valus, c='b', label='KS Test Probability')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Number of Points")
        plt.ylabel("Probability / 1 - p_value")
        plt.title("KS Test For Rand Set: {}".format(i))
        plt.legend(loc='best')
        plt.savefig("plots/KStest_pvalue_{}.png".format(i), dpi=300)
        plt.cla()

one_e(random_generator(5227))

