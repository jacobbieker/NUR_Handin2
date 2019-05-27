import numpy as np
import matplotlib.pyplot as plt


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
    top = np.exp(-x ** 2 / 2)
    bottom = np.sqrt(2 * np.pi)
    return top / bottom


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

    pdf_x = np.linspace(u - 5 * sigma, u + 5 * sigma, 10000)
    pdf_y = gaussian(pdf_x, u, sigma * sigma)

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
