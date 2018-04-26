import sys
import math
import random
from mnist import MNIST
mnist_repo = '../lab2/'

def bin_4to1_convert(x):
    #convert 28*28 to 14*14
    y = []
    for i in range(len(x)):
        yi = []
        for j in range(196):
            idx = int(j/14)*28*2 + (j%14)*2
            p = x[i][idx]+x[i][idx+1]+x[i][idx+28]+x[i][idx+29]

            yi.append(convert(p/4.0))
        y.append(yi)

    return y
    
def convert(x):
    if x > 90:
        return 1
    else:
        return 0

if  __name__ == '__main__':
    
    train_im = open(mnist_repo+'train-images.idx3-ubyte', 'rb')
    train_lb = open(mnist_repo+'train-labels-idx1-ubyte', 'rb')
    test_im = open(mnist_repo+'t10k-images-idx3-ubyte', 'rb')
    test_lb = open(mnist_repo+'t10k-labels-idx1-ubyte', 'rb')
    
    magic_number = int.from_bytes(train_im.read(4), byteorder='big')
    n = int.from_bytes(train_im.read(4), byteorder='big')
    image_row = int.from_bytes(train_im.read(4), byteorder='big')
    image_col = int.from_bytes(train_im.read(4), byteorder='big')
    
    num_of_pixel = image_row * image_col
    
    n = 100

    train_lb.seek(8)
    label = []
    for i in range(n):
        label.append(int.from_bytes(train_lb.read(1), byteorder='big'))
    print(type(label[1]))
    K = 10
    p = [[random.random() for i in range(196)] for j in range(K)]  #p[10][784]
    
    x = []
    

    w = [[random.random() for i in range(K)] for j in range(n)]
    #w = [[1.0 if i == 0 else 0.0 for i in range(K)] for j in range(n)]
    ld = [0.1, 0.2, 0.1, 0.2, 0.05, 0.2, 0.05, 0.07, 0.03, 0.2]
    for i in range(n):
        x.append(train_im.read(784))
    x = bin_4to1_convert(x)
    #for i in range(14):
    #    print(y[2][i*14:i*14+14])
    #t = [0 for i in range(784)]
    #for i in range(784):
    #    t[i] = convert(x[0][i])
    #for i in range(28):
        #print(t[i*28:i*28+28])
    #print(p)
    for a in range(50):
    #E step
        for i in range(n):
            for j in range(K):
                if ld[j] == 0:
                    w[i][j] = -40
                else:
                    w[i][j] = math.log(ld[j])
                for k in range(196):
                    if x[i][k] == 1:
                        if p[j][k] == 0:
                            w[i][j] += -10
                        else:
                            w[i][j] += math.log(p[j][k])
                    else:
                        if p[j][k] == 1:
                            w[i][j] += -10
                        else:
                            w[i][j] += math.log(1 - p[j][k])
            #print(w[i])
            sum_wi = 0 
            for j in range(K):
                sum_wi += math.exp(w[i][j])
            '''
            if sum_wi == 0:
                for j in range(K):
                    w[i][j] = 1.0/K
            else:
            '''
            for j in range(K):
                w[i][j] = math.exp(w[i][j]) / sum_wi
            #print(w[i])
            #print(sum(w[i]))
        print("finish E step")
    #M step
        
        for i in range(K):
            s = 0.0
            for j in range(n):
                s += w[j][i]
            ld[i] = s
            #print(ld[i])
        for i in range(K):
            for j in range(196):
                s = 0.0
                for k in range(n):
                    s += x[k][j]*w[k][i]
                
                p[i][j] = s/ld[i]
            #print(p[i])
        for i in range(K):
            ld[i] = ld[i]/float(n)
        

        print(ld)
        print(sum(ld),a)
        print("finish M step")
    
    # testing
    stat = [[]  for i in range(K)]
    
    for i in range(n):
        stat[w[i].index(max(w[i]))].append(label[i])
    print(stat)

