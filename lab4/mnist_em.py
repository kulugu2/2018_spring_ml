import sys
import math
import random
import copy
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
    y = [[0.0 for i in range(len(x[0]))] for j in range(len(x))]
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] > 150:
                y[i][j] = 1
            else:
                y[i][j] = 0
    return y
        



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
    
    n = 10000

    train_lb.seek(8)
    label = []
    for i in range(n):
        label.append(int.from_bytes(train_lb.read(1), byteorder='big'))
    print(type(label[1]))
    K = 10
    p = [[random.uniform(0.4, 0.6) for i in range(784)] for j in range(K)]  #p[10][784]
    
    x = []
    

    #w = [[random.uniform(0.4, 0.6) for i in range(K)] for j in range(n)]
    w = [[1.0 if i == 0 else 0.0 for i in range(K)] for j in range(n)]
    ld = [0.1, 0.2, 0.1, 0.2, 0.05, 0.2, 0.05, 0.07, 0.03, 0.1]
    for i in range(n):
        x.append(train_im.read(784))
    x = convert(x)
    #x = bin_4to1_convert(x)
    #for i in range(14):
    #    print(y[2][i*14:i*14+14])
    #t = [0 for i in range(784)]
    #for i in range(784):
    #    t[i] = convert(x[0][i])
    #for i in range(28):
        #print(t[i*28:i*28+28])
    a = 0
    while True:
    #E step
        old_p = copy.deepcopy(p)
        for i in range(n):
            for j in range(K):
                if ld[j] == 0:
                    w[i][j] = -40
                else:
                    w[i][j] = math.log(ld[j])
                for k in range(784):
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
            for j in range(784):
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
        converge = 1
        for i in range(K):
            for j in range(784):
                if math.fabs(old_p[i][j] - p[i][j]) > 0.005:
                    converge = 0
                    break
        
        if converge == 1:
            break
        a += 1 
    # testing
    stat = [[0 for i in range(K)]  for j in range(K)]
    
    for i in range(n):
        stat[w[i].index(max(w[i]))][label[i]] += 1
    print(stat)
    
    tp = [0 for i in range(K)]
    tn = [0 for i in range(K)]
    fp = [0 for i in range(K)]
    fn = [0 for i in range(K)]
    kk = [0 for i in range(K)]
    for i in range(K):
        kk[i] = stat[i].index(max(stat[i]))
        tp[i] = stat[i][kk[i]]
        fp[i] = sum(stat[i]) - tp[i]
        for j in range(K):
            fn[i] += stat[j][kk[i]]
        fn[i] -+ tp[i]
        tn[i] = n - fp[i] - fn[i] + tp[i]
    
    sensitivity = [0 for i in range(K)]
    specificity = [0 for i in range(K)]

    for i in range(K):
        sensitivity[i] = tp[i] / float(tp[i]+fn[i])
        specificity[i] = tn[i] / float(tn[i] + fp[i])
    print("kk")
    print(kk)
    print("sensitivity")
    print(sensitivity)
    print("specificity")
    print(specificity)
