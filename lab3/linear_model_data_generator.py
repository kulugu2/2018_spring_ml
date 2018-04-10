import sys
import random
import math
import matplotlib.pyplot as plt

def linear_gen(w, a):

    u = random.random()
    v = random.random()
    
    x = random.random() * 20 - 10
    e = math.sqrt( -2 * math.log(u) ) * math.cos( math.pi * 2 * v)
    #print(a, e)
    e = e*math.sqrt(a) 

    y = 0
    xx = 1
    for ww in w:
        y += ww*xx
        xx *= x

    y += e


    return x, y

if __name__ == '__main__':
    
    basis = int(sys.argv[1])
    a = float(sys.argv[2])
    w = []
    for i in range(basis):
        w.append(float(sys.argv[3+i]))
    x = []
    y = []
    for i in range(1000):
        t1, t2 = linear_gen(w,a)
        x.append(t1)
        y.append(t2)

    plt.scatter(x, y)
    plt.show()
    #y = linear_gen(w, a)
