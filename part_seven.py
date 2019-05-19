import h5py
import numpy as np
import matplotlib.pyplot as plt

class OctNode:
    """Stores the data for an octree node, and spawns its children if possible"""
    def __init__(self, center, size, masses, points, ids, leaves=[]):
        self.center = center                    # center of the node's box
        self.size = size                        # maximum side length of the box
        self.children = []                      # start out assuming that the node has no children

        Npoints = len(points)

        if Npoints == 1:
            # if we're down to one point, we need to store stuff in the node
            leaves.append(self)
            self.COM = points[0]
            self.mass = masses[0]
            self.id = ids[0]
            self.g = np.zeros(3)        # at each point, we will want the gravitational field
        else:
            self.GenerateChildren(points, masses, ids, leaves)     # if we have at least 2 points in the node,
            # spawn its children

            # now we can sum the total mass and center of mass hierarchically, visiting each point once!
            com_total = np.zeros(3) # running total for mass moments to get COM
            m_total = 0.            # running total for masses
            for c in self.children:
                m, com = c.mass, c.COM
                m_total += m
                com_total += com * m   # add the moments of each child
            self.mass = m_total
            self.COM = com_total / self.mass
            #print("{}, {}".format(self.mass, self.COM))

    def GenerateChildren(self, points, masses, ids, leaves):
        """Generates the node's children"""
        octant_index = (points > self.center)  #does all comparisons needed to determine points' octants
        for i in range(2): #looping over the 8 octants
            for j in range(2):
                for k in range(2):
                    in_octant = np.all(octant_index == np.bool_([i,j,k]), axis=1)
                    if not np.any(in_octant): continue           # if no particles, don't make a node
                    dx = 0.5*self.size*(np.array([i,j,k])-0.5)   # offset between parent and child box centers
                    self.children.append(OctNode(self.center+dx,
                                                 self.size/2,
                                                 masses[in_octant],
                                                 points[in_octant],
                                                 ids[in_octant],
                                                 leaves))


class Particle(object):
    def __init__(self, mass, position, velocity, id, node=None):
        self.mass = mass
        self.position = position
        self.node = node
        self.id = id
        self.velocity = velocity


class BHNode:
    def __init__(self, center, length, particles, leaves=[]):
        self.center = center
        self.length = length
        self.children = []

        num_particles = len(particles)

        if num_particles == 1:
            # Single point, so its the leaf
            leaves.append(self)
            self.center_of_mass = np.asarray(particles[0].position)
            self.mass = particles[0].mass
            self.id = particles[0].id
            self.gravity = np.zeros(2)

        else:
            # More than one particle, so need to split up the area
            self.generate_quadrants(particles, leaves)

            total_center_of_mass = np.zeros(2)
            total_mass = 0.
            #print(self.children)
            #print(len(particles))
            for child in self.children:
                m = child.mass
                child_center_of_mass = np.asarray(child.center_of_mass)
                total_mass += m
                total_center_of_mass += (child_center_of_mass * m) # Moment
            if len(self.children) < 1:
                print(len(particles))
                for leaf in particles:
                    m = leaf.mass
                    child_center_of_mass = np.asarray(leaf.position)
                    total_mass += m
                    total_center_of_mass += (child_center_of_mass * m) # Moment
            self.mass = total_mass
            self.center_of_mass = total_center_of_mass / total_mass
            #print(self.center_of_mass)


    def generate_quadrants(self, particles, leaves):
        """
        Generates the leaves if needed
        :param particles:
        :param leaves:
        :return:
        """
        lower_lpart = []
        lower_rpart = []
        upper_rpart = []
        upper_lpart = []

        dx = 0.5*self.length # Change in each direction
        origin = (self.center[0]-dx, self.center[1]-dx)
        lower_left = (origin[0], origin[0]+dx, origin[1], origin[1]+dx) # (x,x,y,y)
        lower_right = (origin[0]+dx, origin[0]+2*dx, origin[1], origin[1]+dx)
        upper_right = (origin[0]+dx, origin[0]+2*dx, origin[1]+dx, origin[1]+2*dx)
        upper_left = (origin[0], origin[0]+dx, origin[1]+dx, origin[1]+2*dx)

        for _, part in enumerate(particles):
            position = part.position
            # To reduce overlap
            if lower_left[0] <= position[0] < lower_left[1] and lower_left[2] <= position[1] < lower_left[3]:
                lower_lpart.append(part)
            elif lower_right[0] <= position[0] <= lower_right[1] and lower_right[2] <= position[1] < lower_right[3]:
                lower_rpart.append(part)
            elif upper_right[0] <= position[0] <= upper_right[1] and upper_right[2] <= position[1] <= upper_right[3]:
                upper_rpart.append(part)
            elif upper_left[0] <= position[0] < upper_left[1] and upper_left[2] <= position[1] <= upper_left[3]:
                upper_lpart.append(part)

        for i in range(2):
            for j in range(2):
                dx = 0.5*self.length*(np.array([i,j])-0.5)   # offset between parent and child box centers
                if (i,j) == (0,0):
                    if len(lower_lpart) > 0:
                        #print(len(lower_lpart))
                        self.children.append(BHNode(self.center+dx,
                                                    self.length/2,
                                                    lower_lpart,
                                                    leaves))
                elif (i,j) == (0,1):
                    if len(lower_rpart) > 0:
                        #print(len(lower_rpart))
                        self.children.append(BHNode(self.center+dx,
                                                    self.length/2,
                                                    lower_rpart,
                                                    leaves))
                elif (i,j) == (1,1):
                    if len(upper_rpart) > 0:
                        #print(len(upper_rpart))
                        self.children.append(BHNode(self.center+dx,
                                                    self.length/2,
                                                    upper_rpart,
                                                    leaves))
                elif (i,j) == (1,0):
                    if len(upper_lpart) > 0:
                        #print(len(upper_lpart))
                        self.children.append(BHNode(self.center+dx,
                                                    self.length/2,
                                                    upper_lpart,
                                                    leaves))


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
    for index in range(len(coords[:])):
        particles.append(Particle(mass=masses[index], position=(coords[index][0],coords[index][1]),
                                  id=ids[index], velocity=velocities[index]))

    # Now have all the particles, add them to the tree
    node = OctNode(masses=masses, points=coords, center=(75,75,75), size=150, ids=ids)
    print(node.mass)
    print(node.COM)
    BHTree = BHNode(center=(75,75), length=150, particles=particles)
    print(BHTree.center_of_mass)
    print(BHTree.mass)

part_seven()

