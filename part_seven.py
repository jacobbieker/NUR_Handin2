import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import sys


class Particle(object):
    def __init__(self, mass, position, velocity, id, node=None):
        self.mass = mass
        self.position = position
        self.node = node
        self.id = id
        self.velocity = velocity


class BHNode:
    def __init__(self, center, length, particles, parent=None):
        self.center = center
        self.length = length
        self.children = []
        self.particles = particles
        self.parent = parent
        self.is_leaf = False
        self.moment = 0

        self.generate_quadrants(limit=12) # Construct the Tree recursively

    def bfs(self):
        all_nodes = [self] + self.children

        for child in self.children:
            all_nodes += child.bfs()

        return all_nodes

    def plot(self, xlabel='', ylabel=''):
        """
        Plots the BH Tree
        :param xlabel:
        :param ylabel:
        :return:
        """
        fig, ax = plt.subplots(1, figsize=(10, 10))
        plt.xlim(self.center[0] - self.length / 2, self.center[0] + self.length / 2)
        plt.ylim(self.center[1] - self.length / 2, self.center[1] + self.length / 2)

        # Get children, so all the leaf nodes
        children = self.get_children()
        for child in children:
            if len(child.particles) > 12: # Check to see if more than 12 particles per leaf node
                print(len(child.particles))

        x = [particle.position[0] for particle in self.particles]
        y = [particle.position[1] for particle in self.particles]

        ax.scatter(x, y, s=1)

        for child in children:
            ax.add_patch(patches.Rectangle((child.center - (child.length / 2)), child.length, child.length, fill=False))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.savefig("plots/bhtree.png", dpi=300)
        plt.cla()
        plt.close()

    def calc_multipole(self):
        for particle in self.particles:
            self.moment += particle.mass

        if not self.is_leaf:
            for child in self.children:
                child.calc_multipole()

    def work_up_tree(self):
        """
        Goes from current node up to the top of the tree, printing out the stuff along the way
        :return:
        """
        print("At (X,Y): {} Size: {} n = 0 moment: {}".format(self.center, self.length, self.moment))
        if self.parent is not None:
            self.parent.work_up_tree()  # Go up to the next level

    def get_children(self):
        """
        Gets all the children of this node and returns them, includes all children of children

        On the root node, it would return all children in the tree

        :return:
        """
        children = []

        if self.is_leaf:
            children.append(self)

        else:
            for child in self.children:
                children += child.get_children()

        return children

    def get_particle(self, id):
        """
        Goes through all particles, getting the node with that ID
        :param id:
        :return:
        """

        for i in range(len(self.particles)):
            if self.particles[i].id == id:
                if self.particles[i].node.is_leaf:
                    # Start here and work way back up
                    print("At Node {}: (X,Y): {} n = 0 moment: {}".format(self.particles[i].id,
                                                                          self.particles[i].position,
                                                                          self.particles[i].node.moment))
                    # Now work back up to top of tree
                    self.particles[i].node.work_up_tree()

    def generate_quadrants(self, limit=12):
        """
        Generates the leaves if needed
        :param particles:
        :param leaves:
        :return:
        """
        if len(self.particles) == 0:
            self.is_leaf = True
            return

        elif len(self.particles) < limit:
            self.is_leaf = True
            for part in self.particles:
                part.node = self
            return

        elif len(self.particles) >= limit:
            lower_lpart = []
            lower_rpart = []
            upper_rpart = []
            upper_lpart = []
            for part in self.particles:
                part.node = self

            dx = 0.5 * self.length  # Change in each direction
            origin = (self.center[0] - dx, self.center[1] - dx) # Origin for bottom left one
            lower_left = (origin[0], origin[0] + dx, origin[1], origin[1] + dx)  # (x,x,y,y)

            lower_right = (origin[0] + dx, origin[0] + 2 * dx, origin[1], origin[1] + dx)

            upper_right = (origin[0] + dx, origin[0] + 2 * dx, origin[1] + dx, origin[1] + 2 * dx)

            upper_left = (origin[0], origin[0] + dx, origin[1] + dx, origin[1] + 2 * dx)

            for part in self.particles: # Goes through and puts the particles in each correct quadrant
                position = part.position
                if lower_left[0] <= position[0] <= lower_left[1] and lower_left[2] <= position[1] <= lower_left[3]:
                    lower_lpart.append(part)
                elif lower_right[0] <= position[0] <= lower_right[1] and lower_right[2] <= position[1] <= lower_right[3]:
                    lower_rpart.append(part)
                elif upper_right[0] <= position[0] <= upper_right[1] and upper_right[2] <= position[1] <= upper_right[
                    3]:
                    upper_rpart.append(part)
                elif upper_left[0] <= position[0] <= upper_left[1] and upper_left[2] <= position[1] <= upper_left[3]:
                    upper_lpart.append(part)

            for i in range(2):
                for j in range(2):
                    dx = 0.5 * self.length * (np.array([i, j]) - 0.5)  # offset between parent and child box centers
                    if (i, j) == (0, 0):
                        self.children.append(BHNode(center=self.center + dx,
                                                    length=self.length / 2,
                                                    particles=lower_lpart,
                                                    parent=self))
                    elif (i, j) == (0, 1):
                        self.children.append(BHNode(center=self.center + dx,
                                                    length=self.length / 2,
                                                    particles=lower_rpart,
                                                    parent=self))
                    elif (i, j) == (1, 1):
                        self.children.append(BHNode(center=self.center + dx,
                                                    length=self.length / 2,
                                                    particles=upper_rpart,
                                                    parent=self))
                    elif (i, j) == (1, 0):
                        self.children.append(BHNode(center=self.center + dx,
                                                    length=self.length / 2,
                                                    particles=upper_lpart,
                                                    parent=self))

        return self

# Now run the Part 7 stuff
def part_seven():
    sys.stdout = open('7.txt', 'w')
    with h5py.File('colliding.hdf5', 'r') as f:
        f = f['PartType4']
        coords = f['Coordinates'][:]
        masses = f['Masses'][:]
        ids = f['ParticleIDs'][:]
        velocities = f['Velocities'][:]

    # Create Particles Only care about 2D in this example
    particles = []
    for index in range(len(coords[:])):
        particles.append(Particle(mass=masses[index], position=(coords[index][0], coords[index][1]),
                                  id=ids[index], velocity=velocities[index]))

    # Now have all the particles, add them to the tree
    BHTree = BHNode(center=(75, 75), length=150, particles=particles)
    BHTree.calc_multipole()

    id_100 = particles[100].id

    BHTree.get_particle(id_100)
    BHTree.plot("X Coordinate", "Y Coordinate")