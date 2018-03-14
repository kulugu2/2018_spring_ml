import matplotlib.pyplot as plt
import numpy as np
import sys

def show(X, Y, l1, l2):
    #X = []
    #Y = []
    #fp = open(sys.argv[1], 'r')
    #for line in iter(fp):
    #    l = line.strip().split(',')
    #    X.append(float(l[0]))
    #    Y.append(float(l[1]))
    x1 = np.linspace(min(X),max(X),100)
    y1 = 0
    for i in range(len(l1)):
        y1 += l1[i]*x1**i  
    
    y2 = 0
    for i in range(len(l2)):
        y2 += l2[i]*x1**i

    plt.plot(X,Y,"o")
    plt.plot(x1,y1)
    plt.plot(x1,y2)
    plt.show()
