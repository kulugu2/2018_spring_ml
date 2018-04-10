import sys
import random
import math
import matplotlib.pyplot as plt
import univariate_gaussian_data_generator as gen

if __name__ == '__main__':
    
    input_m = float(sys.argv[1])
    input_s = float(sys.argv[2])
    
    new_x = gen.univariate_gen(input_m, input_s)
    print(new_x)
    
    m = new_x
    m2 = 0.0
    n = 1
    s = 0.0

    print("m = ",m, "s = ", s)
    input()

    while n<10000:
        new_x = gen.univariate_gen(input_m, input_s)
        print(new_x)
        n += 1
        old_m = m
        m = m + (new_x - m) / float(n)
        m2 = m2 + (new_x - old_m) * (new_x - m)
        s = m2 / float(n)
        print("m = ",m, "s = ", s)
        #input()

