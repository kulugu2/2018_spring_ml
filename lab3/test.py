import sys
import random
import math
import matplotlib.pyplot as plt

def univariate_gen(m, s):

    u = random.random()
    v = random.random()
    
    x = math.sqrt( -2 * math.log(u) ) * math.cos( math.pi * 2 * v)
    x = x*math.sqrt(s) + m
    return x
    
if __name__ == '__main__':
    #m = float(sys.argv[1])
    #s = float(sys.argv[2])

    ''' for testing 
    x = []
    for i in range(10000):
        x.append( univariate_gen(m, s) )
    
    count = 0
    for l in x:
        if l < m+1*math.sqrt(s) and l > m-1*math.sqrt(s):
            count += 1
    print(count)
    plt.hist(x)
    plt.show() 
    '''
    x = []
    print(math.cos(0), math.sin(0),math.cos(math.pi))
    for i in range(10000):
        x.append(math.sin(random.random() *2*  math.pi))
    plt.hist(x, bins=100)
    plt.show()

    
    

