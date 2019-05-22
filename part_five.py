import numpy as np
import matplotlib.pyplot as plt


def nearest_grid_point_method(grid, mass, locations):
    """
    Performs Nearest Grid Point Method for a given grid, with mass distributed at locations



    :param grid:
    :param mass:
    :param locations:
    :return:
    """

def get_closest_point(grid, position):
    """
    Calculates the closest grid point to the current one
    :param grid:
    :param position:
    :return:
    """

    lower = []
    upper = []
    for pos in position:
        lower.append(int(pos))
        upper.append(int(pos)+1) # Ceiling





def part_five_a(rand_gen):
    """
    Point like Nearest Grid Point method

    Shape is given by S(x) = 1/(cell size) delta(position/cell size)

    NGP method assigns the particls mass to the closest point

    :param rand_gen:
    :return:
    """

    np.random.seed(121)
    positions = np.random.uniform(low=0,high=16,size=(3,1024))

    grid = np.zeros((16,16,16))
    grid[:,:,4]

    # Now need to do the NGP method
    # Need to find the closest point in each dimension for each postion, take from earlier
    # C

    # Now need to do 
