import h5py
import numpy as np
import matplotlib.pyplot as plt


class OctNode:
    """Stores the data for an octree node, and spawns its children if possible"""

    def __init__(self, center, size, masses, points, ids, leaves=[]):
        self.center = center  # center of the node's box
        self.size = size  # maximum side length of the box
        self.children = []  # start out assuming that the node has no children

        Npoints = len(points)

        if Npoints == 1:
            # if we're down to one point, we need to store stuff in the node
            leaves.append(self)
            self.COM = points[0]
            self.mass = masses[0]
            self.id = ids[0]
            self.g = np.zeros(3)  # at each point, we will want the gravitational field
        else:
            self.GenerateChildren(points, masses, ids, leaves)  # if we have at least 2 points in the node,
            # spawn its children

            # now we can sum the total mass and center of mass hierarchically, visiting each point once!
            com_total = np.zeros(3)  # running total for mass moments to get COM
            m_total = 0.  # running total for masses
            for c in self.children:
                m, com = c.mass, c.COM
                m_total += m
                com_total += com * m  # add the moments of each child
            self.mass = m_total
            self.COM = com_total / self.mass
            # print("{}, {}".format(self.mass, self.COM))

    def GenerateChildren(self, points, masses, ids, leaves):
        """Generates the node's children"""
        octant_index = (points > self.center)  # does all comparisons needed to determine points' octants
        for i in range(2):  # looping over the 8 octants
            for j in range(2):
                for k in range(2):
                    in_octant = np.all(octant_index == np.bool_([i, j, k]), axis=1)
                    if not np.any(in_octant): continue  # if no particles, don't make a node
                    dx = 0.5 * self.size * (np.array([i, j, k]) - 0.5)  # offset between parent and child box centers
                    self.children.append(OctNode(self.center + dx,
                                                 self.size / 2,
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
    def __init__(self, center, length, particles, parent=None):
        self.center = center
        self.length = length
        self.children = []
        self.points = particles
        self.parent = parent
        self.is_leaf = False
        self.moment = 0

        self.generate_quadrants(limit=12)

    def plot(self, xlabel='', ylabel='', filename="plots/bhtree.png", save=False):
        fig, ax = plt.subplots(1, figsize=(7, 7))
        plt.xlim(self.center[0] - self.length / 2, self.center[0] + self.length / 2)
        plt.ylim(self.center[1] - self.length / 2, self.center[1] + self.length / 2)

    def calc_multipole(self):
        for point in self.points:
            self.moment += point.mass

        if not self.is_leaf:
            for child in self.children:
                child.calc_multipole()

    def work_up_tree(self):
        """
        Goes from current node up to the top of the tree
        :return:
        """
        print("At (X,Y): {} Size: {} n = 0 moment: {}".format(self.center, self.length, self.moment))
        if self.parent is not None:
            self.parent.work_up_tree() # Go up to the next level


    def get_point(self, id):
        """
        Want to do Breadth First
        :param id:
        :return:
        """

        for i in range(len(self.points)):
            if self.points[i].id == id:
                if self.points[i].node.is_leaf:
                    # Start here and work way back up
                    print("At Node {}: (X,Y): {} n = 0 moment: {}".format(self.points[i].id, self.points[i].position,
                                                                            self.points[i].node.moment))
                    # Now work back up to top of tree
                    self.points[i].node.work_up_tree()

    def generate_quadrants(self, limit=12):
        """
        Generates the leaves if needed
        :param particles:
        :param leaves:
        :return:
        """
        if len(self.points) == 0:
            self.is_leaf = True
            return

        elif int(len(self.points)) <= limit:
            self.is_leaf = True
            for part in self.points:
                part.node = self
            return

        elif len(self.points) > limit:
            lower_lpart = []
            lower_rpart = []
            upper_rpart = []
            upper_lpart = []

            dx = 0.5 * self.length  # Change in each direction
            origin = (self.center[0] - dx, self.center[1] - dx)
            lower_left = (origin[0], origin[0] + dx, origin[1], origin[1] + dx)  # (x,x,y,y)
            lower_right = (origin[0] + dx, origin[0] + 2 * dx, origin[1], origin[1] + dx)
            upper_right = (origin[0] + dx, origin[0] + 2 * dx, origin[1] + dx, origin[1] + 2 * dx)
            upper_left = (origin[0], origin[0] + dx, origin[1] + dx, origin[1] + 2 * dx)

            for _, part in enumerate(self.points):
                position = part.position
                # To reduce overlap
                if lower_left[0] <= position[0] < lower_left[1] and lower_left[2] <= position[1] < lower_left[3]:
                    lower_lpart.append(part)
                elif lower_right[0] <= position[0] <= lower_right[1] and lower_right[2] <= position[1] < lower_right[3]:
                    lower_rpart.append(part)
                elif upper_right[0] <= position[0] <= upper_right[1] and upper_right[2] <= position[1] <= upper_right[
                    3]:
                    upper_rpart.append(part)
                elif upper_left[0] <= position[0] < upper_left[1] and upper_left[2] <= position[1] <= upper_left[3]:
                    upper_lpart.append(part)

            for i in range(2):
                for j in range(2):
                    dx = 0.5 * self.length * (np.array([i, j]) - 0.5)  # offset between parent and child box centers
                    #print(self.center + dx)
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
                                                    length= self.length / 2,
                                                    particles=upper_lpart,
                                                    parent=self))

            for child in self.children:
                if len(child.points) > limit:
                    child.generate_quadrants(limit=12)

        return self


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
    print(len(coords))
    for index in range(len(coords[:])):
        particles.append(Particle(mass=masses[index], position=(coords[index][0], coords[index][1]),
                                  id=ids[index], velocity=velocities[index]))

    # Now have all the particles, add them to the tree
    node = OctNode(masses=masses, points=coords, center=(75, 75, 75), size=150, ids=ids)
    print(node.mass)
    print(node.COM)
    BHTree = BHNode(center=(75, 75), length=150, particles=particles)
    BHTree.calc_multipole()
    print(BHTree.moment)

    id_100 = particles[100].id

    BHTree.get_point(id_100)


part_seven()
