from one import random_generator, all_one
from part_two import part_two
from part_three import part_three
from four import all_four
from part_six import part_six
from part_seven import part_seven
import matplotlib.pyplot as plt

import time
import sys



start = time.time()
sys.stdout = open('seed.txt', 'w')


seed = 5227

print("The seed for this project is: {}".format(seed))

rand_gen = random_generator(seed)

all_one(rand_gen)
end = time.time()
sys.stdout = open('end.txt', 'w')
print("Elapsed Time: {}".format(end - start))

plt.cla()
part_two(rand_gen)
end = time.time()
sys.stdout = open('end.txt', 'a')
print("Elapsed Time: {}".format(end - start))
plt.cla()

# Now do multiprocessing as these can take awhile
part_three()
plt.cla()
end = time.time()
sys.stdout = open('end.txt', 'a')
print("Elapsed Time: {}".format(end - start))
all_four()
plt.cla()
end = time.time()
sys.stdout = open('end.txt', 'a')
print("Elapsed Time: {}".format(end - start))
#part_six()
end = time.time()
sys.stdout = open('end.txt', 'a')
print("Elapsed Time: {}".format(end - start))
plt.cla()
part_seven()
end = time.time()
sys.stdout = open('end.txt', 'a')
print("Elapsed Time: {}".format(end - start))
plt.cla()

end = time.time()

sys.stdout = open('end.txt', 'w')
print("Elapsed Time: {}".format(end - start))
