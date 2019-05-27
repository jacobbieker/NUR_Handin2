from one import random_generator, all_one
from part_two import part_two
from part_three import part_three
from four import all_four
from part_six import part_six
from part_seven import part_seven

import sys

sys.stdout = open('seed.txt', 'w')


seed = 5227

print("The seed for this project is: {}".format(seed))

rand_gen = random_generator(seed)

all_one(rand_gen)

part_two(rand_gen)
part_three()
all_four()
part_six()
part_seven()

