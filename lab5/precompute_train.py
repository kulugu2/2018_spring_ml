import sys
from svm import *
from svmutil import *
import math
gamma = 0.001
def dot(a, b):
    result = 0
    for key, value in a.iteritems():
        if b.has_key(key):
            result += value*b[key]
    return result

file_name = 'train_libsvm_format'
#file_name = 'test'
output_file = 'precompute_libsvm_format'

if __name__ == '__main__':

    y, x = svm_read_problem(file_name)
    
    output = open(output_file, 'w')
    print(type(x[0])) 
    n = len(y)
    xx = []
    for i in range(n):
        xx.append(dot(x[i],x[i]))
    for i in range(n):
        print(i)
        output.write(str(int(y[i])))
        output.write(' 0:')
        output.write(str(i+1))
        output.write(' ')

        x2 = xx[i]
        for j in range(n):
            xy = dot(x[i], x[j])
            y2 = xx[j]
            rbf = math.exp(-gamma*(x2 - 2*xy + y2))
            output.write(str(j+1)+':'+ str('%.3f' % (rbf + xy))+' ')
        output.write('\n')


