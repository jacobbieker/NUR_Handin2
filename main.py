from one import random_generator, all_one
from part_two import part_two
from part_three import part_three
from four import all_four
from part_six import part_six
from part_seven import part_seven
import matplotlib.pyplot as plt
import multiprocessing

import time
import sys



start = time.time()
sys.stdout = open('seed.txt', 'w')


seed = 5227

print("The seed for this project is: {}".format(seed))

rand_gen = random_generator(seed)

all_one(rand_gen)
end = time.time()
sys.stdout = open('mend.txt', 'w')
print("Elapsed Time: {}".format(end - start))

plt.cla()
part_two(rand_gen)
end = time.time()
sys.stdout = open('mend.txt', 'a')
print("Elapsed Time: {}".format(end - start))
plt.cla()

# Now do multiprocessing as these can take awhile, and don't rely on the RNG
procs = []

proc = multiprocessing.Process(target=part_six)
procs.append(proc)
proc.start()
proc = multiprocessing.Process(target=part_three)
procs.append(proc)
proc.start()
proc = multiprocessing.Process(target=part_seven)
procs.append(proc)
proc.start()
proc = multiprocessing.Process(target=all_four)
procs.append(proc)
proc.start()

#part_three()
#plt.cla()
#end = time.time()
#sys.stdout = open('end.txt', 'a')
#print("Elapsed Time: {}".format(end - start))
#all_four()
#plt.cla()
#end = time.time()
#sys.stdout = open('end.txt', 'a')
#print("Elapsed Time: {}".format(end - start))
#part_six()
#end = time.time()
#sys.stdout = open('end.txt', 'a')
#print("Elapsed Time: {}".format(end - start))
#plt.cla()
#part_seven()
#end = time.time()
#sys.stdout = open('end.txt', 'a')
#print("Elapsed Time: {}".format(end - start))
#plt.cla()

for proc in procs:
    proc.join() # Wait for them to all join
end = time.time()

sys.stdout = open('mend.txt', 'a')
print("Elapsed Time: {}".format(end - start))
