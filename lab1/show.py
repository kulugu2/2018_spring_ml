import matplotlib.pyplot as plt
import numpy as np
import sys

X = []
Y = []
fp = open(sys.argv[1], 'r')
for line in iter(fp):
    l = line.strip().split(',')
    X.append(float(l[0]))
    Y.append(float(l[1]))


plt.plot(X,Y,"o")
plt.show()
