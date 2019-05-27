from one import random_generator, all_one
from part_two import part_two
from part_three import part_three
from four import all_four
from part_six import part_six
from part_seven import part_seven

import time
import sys

start = time.time()
sys.stdout = open('seed.txt', 'w')


seed = 5227

print("The seed for this project is: {}".format(seed))

rand_gen = random_generator(seed)

print("Start All One")
all_one(rand_gen)
print("End All One")
part_two(rand_gen)
print("End Part 2")

# Now do multiprocessing as these can take awhile
part_three()
print("End Part 3")
all_four()
print("End Part 4")
#part_six()
part_seven()
print("End Part 7")

end = time.time()

sys.stdout = open('end.txt', 'w')
print("Elapsed Time: {}".format(end - start))
