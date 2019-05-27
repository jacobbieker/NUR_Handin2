import numpy as np
import matplotlib.pyplot as plt

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
