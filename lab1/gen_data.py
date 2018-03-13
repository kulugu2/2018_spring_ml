
from random import random
import sys

n = int(sys.argv[1])
b = int(sys.argv[2])
p = [float(i) for i in sys.argv[3:]]
p.reverse()
for i in range(n):
        x = random() * n * 1
        y = sum([pi * x**j for j, pi in enumerate(p)]) + (random()-0.5)
        print "%f,%f" % (x, y)
