import h5py
import numpy as np
import matplotlib.pyplot as plt

total_inserts = 0
class Node(object):
    def __init__(self, parent=None, children=[], mass=0, x_range=(0,0),
                 y_range=(0,0), is_leaf=True, particle_set=None):
        self.parent = parent
        self.children = children
        self.center_of_mass = (-1,-1)
        self.mass = mass
        self.multipole_moment = -1
        self.x_range = x_range
        self.y_range = y_range
        self.is_leaf = is_leaf
        self.particle_set = particle_set


class Particle(object):
    def __init__(self, mass, position, velocity, id, node=None):
        self.mass = mass
        self.position = position
        self.node = node
        self.id = id
        self.velocity = velocity


class QuadTree(object):
    def __init__(self, xlim, ylim):
        self.center_of_mass = 0
        self.multipole_moment = 0
        self.mass = 0
        self.root = Node(x_range=xlim, y_range=ylim)

    def make_quadrants(self, node):
        x_len = (node.x_range[1] - node.x_range[0])/2
        y_len = (node.y_range[1] - node.y_range[0])/2

        # Make the 4 Quadrants

        lower_right = Node(parent=node, children=[],
                           x_range=(node.x_range[0],node.x_range[0]+x_len),
                           y_range=(node.y_range[0],node.y_range[0]+y_len))

        lower_left = Node(parent=node, children=[],
                          x_range=(node.x_range[0]+x_len,node.x_range[1]),
                          y_range=(node.y_range[0],node.y_range[0]+y_len))

        upper_right = Node(parent=node, children=[],
                           x_range=(node.x_range[0],node.x_range[0]+x_len),
                           y_range=(node.y_range[0]+y_len,node.y_range[1]))

        upper_left = Node(parent=node, children=[],
                          x_range=(node.x_range[0]+x_len,node.x_range[1]),
                          y_range=(node.y_range[0]+y_len,node.y_range[1]))

        node.children.append(lower_right)
        node.children.append(lower_left)
        node.children.append(upper_left)
        node.children.append(upper_right)

        return node

    def add_particle(self, particle, node=None):
        """
        Recursively add particles to the correct node in the BHTree
        :param mass:
        :param position:
        :return:
        """

        if node is None:
            node = self.root

        if node.is_leaf and node.center_of_mass == (-1,-1): # Base case, leaf node, nothing is stored here
            # no children, so add here, area is 1/4th the area of the current node

            # Since no children, center of mass and mass are just that of the particle
            node.center_of_mass = particle.position
            node.mass = particle.mass
            node.particle_set = (particle,)

            # Return from the leaf
            return node.mass, node.center_of_mass

        else:
            # There are children so need to split into 4 Quadrants
            masses = []
            positions = []
            particles = [particle]
            if len(node.children) == 0:
                # Need to make the quadrants once, but only if things need to be made smaller
                # Once its not a leaf node, then pop all off of the partcle set, and add them one by one
                node = self.make_quadrants(node)
                node.is_leaf = False
            # Now need to move the current particle into a quadrant, then move the new particle to its quadrant
            # Recursively
            if len(node.particle_set) > 0:
                for p in node.particle_set:
                    particles.append(p)
                node.particle_set = ()

            # Add node particle to
            for p in particles:
                x_pos = particle.position[0]
                y_pos = particle.position[1]
                for child in node.children:
                    if child.x_range[0] <= x_pos <= child.x_range[1]:
                        if child.y_range[0] <= y_pos <= child.y_range[1]:
                            # In this quadrant, adds the particle
                            m, com = self.add_particle(particle, child)
                            masses.append(m)
                            positions.append(com)

            # Now get the center of mass and mass of the current node, need to add them up by the weighted values
            total_mass = np.sum(masses)
            tmp_x = 0
            tmp_y = 0
            for index in range(len(masses)):
                tmp_x += masses[index]*positions[index][0] # Weighted be the mass
                tmp_y += masses[index]*positions[index][1]

            center_of_mass = (tmp_x/total_mass, tmp_y/total_mass)

            node.mass = total_mass
            node.center_of_mass = center_of_mass

            return node.mass, node.center_of_mass

    def calc_multipole(self, node):
        """
        Calculates the multipole moment of n=0 for the given node

        Multipole_0 moment is the sum of the masses?


        :param node:
        :return:
        """

    def plot_map(self):
        """
        Plots the BHTree with the particles
        :return:
        """




# Now run the Part 7 stuff
def part_seven():
    with h5py.File('colliding.hdf5', 'r') as f:
        f = f['PartType4']
        coords = f['Coordinates'][:]
        masses = f['Masses'][:]
        ids = f['ParticleIDs'][:]
        velocities = f['Velocities'][:]

    # Create Particles Only care about 2D in this example
    particles = []
    for index in range(len(coords[0:2])):
        particles.append(Particle(mass=masses[index], position=(coords[index][0],coords[index][1]),
                                  id=ids[index], velocity=velocities[index]))

    # Now have all the particles, add them to the tree
    BHTree = QuadTree(xlim=(0,150), ylim=(0,150))
    for particle in particles:
        BHTree.add_particle(particle, BHTree.root)
        #print(BHTree.root.center_of_mass)
        print(BHTree.root.mass)


part_seven()

