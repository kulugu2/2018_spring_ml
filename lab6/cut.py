import sys
import numpy as np
import matplotlib.pyplot as plt
from pca import pca
from sklearn.cluster import KMeans
import math
import random
from datetime import datetime
def read_csv(fn):
    A = []
    with open(fn, 'r') as fp:
        for line in fp.readlines():
            A.append(line.strip().split(','))
    return A
def compute_kernel(data, kernel):
    A = []
    n = len(data)
    k = len(data[0])
    if kernel == 'linear':
        for i in range(n):
            row = []
            for j in range(n):
                tmp = 0
                for kk in range(k):
                    if data[i][kk] == 0 or data[j][kk] == 0:
                        continue
                    else:
                        tmp += data[i][kk] * data[j][kk]
                row.append(tmp)
            A.append(row)
            print(i)
    return A
def computeU(evalue, evector, n):
    U = np.empty(shape=(5000,0))
    ind = np.argmin(evalue)
    print(evector[:, ind])
    print(evalue[ind])
    evalue = np.delete(evalue, ind)
    evector = np.delete(evector, ind, axis = 1)
    
    for i in range(n):
        
        ind = np.argmin(evalue)
        #U = np.append(U, [evector[:, ind]], axis = 1)
        U = np.c_[U, evector[:, ind]]
        print(evector[:, ind])
        print(evalue[ind])
        #U.append(evector[:,ind])
        evalue = np.delete(evalue,ind)
        evector = np.delete(evector, ind, axis =1)
    return U

def distance(a, b):
    dis = 0
    for i in range(len(a)):
        dis += (a[i]-b[i])**2
    return dis 
        
def kmeans(data, k):
    center = []
    n = len(data)
    print(n)
    random.seed(datetime.now())
    for i in range(k):
        center.append(data[int(i*1000)+800])
        #center.append(data[int(random.random()*n)])
    label = []
    for i in range(n):
        mindis = sys.float_info.max
        c = 0
        for j in range(k):
            dis = distance(data[i], center[j])
            if(dis < mindis):
                mindis = dis
                c = j
        label.append(c)
    
    # start clustering

    for _ in range(100):
        num = [0 for i in range(k)]
        center_sum = [[0.0 for i in range(len(data[0]))] for j in range(k)]

        for i in range(n):
            num[label[i]] += 1
            for j in range(len(data[0])):
                center_sum[label[i]][j] += data[i][j]

        for i in range(k):
            for j in range(len(data[0])):
                center[i][j] = center_sum[i][j]/num[i]

        
        for i in range(n):
            mindis = sys.float_info.max
            c = 0
            for j in range(k):
                dis = distance(data[i], center[j])
                if(dis < mindis):
                    mindis = dis
                    c = j
            label[i] = c


    return label
                


color = ['r', 'g', 'b', 'y', 'c']
if __name__ == '__main__':
    data_point = read_csv('X_train.csv')
    data_point = np.array(data_point).astype(np.float)
     
    #W1 = read_csv('linear_precompute')
    #W1 = np.array(W1).astype(np.float)
    #print(W1.shape)
    wfp = open('linear_precompute_byte', 'rb')
    buf = wfp.read(5000*5000*8)
    wfp.close()
    W1 = np.frombuffer(buf, dtype=np.float64)
    W1 = np.reshape(W1, (5000, 5000))
    wfp = open('rbf_precompute_byte', 'rb')
    buf = wfp.read(5000*5000*8)
    wfp.close()
    W = np.frombuffer(buf, dtype=np.float64)
    W = np.reshape(W, (5000, 5000))
    n = len(W)
    #print(W)
    #print(len(W[0]))
    #D = np.sum(W, axis = 1)
    WW =  W1 + W
    D = np.sum(WW, axis = 1)
    WW = WW*-1
    for i in range(n):
        WW[i][i] += D[i]
    DD = np.zeros((n,n), dtype=float)
    for i in range(n):
        DD[i][i] = D[i]**-0.5
    print(np.matmul(DD,WW))
    nomalize = np.matmul(np.matmul(DD, WW), DD)
    #print(nomalize)
    evalue, evector = np.linalg.eig(WW)
    #evalue, evector = np.linalg.eig(nomalize)
    wfp = open('l+r_evalue_ratio', 'wb')
    wfp.write(evalue.real.tobytes())
    wfp.close()
    wfp = open('l+r_evector_ratio', 'wb')
    wfp.write(evector.real.tobytes())
    wfp.close()
    
    '''    
    wfp = open('l+r_evalue_ratio', 'rb')
    buf = wfp.read(5000*8*2)
    evalue = np.frombuffer(buf, dtype=np.float64)
    #print(evalue.shape)
    #print(min(evalue))
    wfp.close()
    wfp = open('l+r_evector_ratio', 'rb')
    buf = wfp.read(5000*5000*8*2)
    evector = np.frombuffer(buf, dtype=np.float64)
    evector = np.reshape(evector, (5000, 5000))
    print(np.sort(evalue))
    '''

    U = computeU(evalue, evector, 5)
    print(U)
    
    for i in range(len(U)):
        divisor = 0
        for j in range(len(U[0])):
            divisor += U[i][j]**2
        divisor = math.sqrt(divisor)
        for j in range(len(U[0])):
            U[i][j] /= divisor
    
    label = kmeans(U, 5)
    num = [0 for i in range(5)]
    for i in label:
        num[i] += 1
    print(num)

    kmean = KMeans(n_clusters = 5,  init='random').fit(U)
    print(kmean.labels_)
    num = [0 for i in range(5)]
    for i in kmean.labels_:
        num[i] += 1
    print(num)
    pca(data_point, label, 'ratio_linear_rbf.png')

