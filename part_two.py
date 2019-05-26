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


def gauss_complex(rand_gen, num_samples, u=0, sigma=1):
    real = box_muller(rand_gen, num_samples=num_samples)
    imaginary = box_muller(rand_gen, num_samples=num_samples)
    real = map_to_guass(real, u=u, sigma=sigma)
    imaginary = map_to_guass(imaginary, u=u, sigma=sigma)
    complex_gauss = real + imaginary * complex(0, 1)
    return complex_gauss


def power_spectrum(kx, ky, n):
    """
    Calcs powerspecturm, which is proportional of k**n = (kx,ky)**n
    :param kx:
    :param ky:
    :param n:
    :return:
    """
    if kx == 0 and ky == 0:
        return 0
    else:
        return np.sqrt(kx ** 2 + ky ** 2) ** n


def part_two(rand_gen):
    """
    Make an initial density field

    :param rand_gen:
    :return:
    """
    grid_size = 1024
    # Create kx, ky grid points lists
    kx = [i for i in range(-511, 512, step=1)]
    ky = [i for i in range(-511, 512, step=1)]
    amplitudes = np.zeros((grid_size, grid_size))

    # Now for each of the n's
    for n in [-1, -2, -3]:
        random_field = np.fft.fft2(box_muller(rand_gen, grid_size ** 2).reshape((grid_size, grid_size)))

        # Make the amplitude matrix
        for i in range(int(grid_size)):
            for j in range(int(grid_size)):
                amplitudes[j + i * grid_size] = power_spectrum(kx[i], ky[j], n)

        # convert to what we want
        random_field = random_field.reshape(-1)
        random_field = random_field * amplitudes
        # Reshape back to the grid and do the fft
        random_field = np.fft.ifft2(random_field.reshape((grid_size, grid_size)))

        im = plt.matshow(np.absolute(random_field))
        plt.xlabel("x (kpc)")
        plt.ylabel("y (kpc)")
        plt.title("Gaussian Random Field n = {}".format(n))
        plt.colorbar(im, label="Absolute value of k")
        plt.savefig("plots/GaussianField_{}.png".format(n), dpi=300)
        plt.cla()