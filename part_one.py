import numpy as np
import matplotlib.pyplot as plt

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

        seed = mwc_out # set the seed to a new number, so a different number generated next time
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

    x1 = np.cos(2*np.pi*z2) * np.sqrt((-2)*np.log(z1))
    return x1


def map_to_guass(x, u, sigma):

    # First change variance

    x = x * sigma

    # then change the mean
    x = x + u

    return x

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

    # TODO Get the actual Guassian Distribution

    plt.hist(gauss, density=True, bins=50)
    plt.axvline(x=u+sigma, c='r')
    plt.axvline(x=u-sigma, c='r')
    plt.axvline(x=u-2*sigma, c='r')
    plt.axvline(x=u-3*sigma, c='r')
    plt.axvline(x=u-4*sigma, c='r')
    plt.axvline(x=u+2*sigma, c='r')
    plt.axvline(x=u+3*sigma, c='r')
    plt.axvline(x=u+4*sigma, c='r')
    plt.xlim(u-5*sigma, u+5*sigma)
    plt.savefig("./plots/box_gauss.png", dpi=300)
    plt.cla()


def one_c(rand_gen):
    """
    KS Test
    :param rand_gen:
    :return:
    """
    sigma = 1
    u = 0
    gauss = box_muller(rand_gen, 1000)
    gauss = map_to_guass(gauss, u=u, sigma=sigma)



