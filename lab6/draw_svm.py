import sys
import numpy as np
import matplotlib.pyplot as plt
from svm import *
from svmutil import *
def read_csv(fn):
    A = []
    with open(fn, 'r') as fp:
        for line in fp.readlines():
            A.append(line.strip().split(','))
    return A
def pca(data, label):
    color = ['r', 'g', 'b', 'y', 'c']
    dataT = np.transpose(data)
    cov_mat = np.cov(dataT)
    
    evalue, evector = np.linalg.eig(cov_mat)

    w = []
    ind = np.argmax(evalue)
    
    w.append(evector[:,ind])
    evalue = np.delete(evalue,ind)
    evector = np.delete(evector, ind, axis =1)

    ind = np.argmax(evalue)
    w.append(evector[:,ind])
    w = np.array(w)     #[2, 784]
    new_data = np.matmul(data, w.T)
    
    for i, element in enumerate(new_data.real):
        ci = int(i/1000)
        plt.scatter(element[0], element[1], c = color[label[i]])
    plt.show()

color = ['r', 'g', 'b', 'y', 'c']
if __name__ == '__main__':
    data = read_csv('X_train.csv')
    data = np.array(data).astype(np.float)
    label = []
    for i in range(5000):
        label.append(int(i/1000))

    #pca(data, label)
    
    dataT = np.transpose(data)
    cov_mat = np.cov(dataT)

    evalue, evector = np.linalg.eig(cov_mat)
    w = []
    ind = np.argmax(evalue)
    
    w.append(evector[:,ind])
    evalue = np.delete(evalue,ind)
    evector = np.delete(evector, ind, axis =1)
    ind = np.argmax(evalue)
    w.append(evector[:,ind])
    w = np.array(w)     #[2, 784]
    new_data = np.matmul(data, w.T)

    #    compute sv

    y, x = svm_read_problem('train_libsvm_format')
    prob = svm_problem(y, x)
    param = svm_parameter('-t 2 -c 4 -b 0')
    param_poly = svm_parameter('-t 1 -g 1')
    param_linear = svm_parameter('-t 0 -c 0.1')
    model = svm_train(prob, param_poly)
    sv = model.get_sv_indices()
    for i, element in enumerate(new_data.real):
        ci = int(i/1000)
        if i+1 in sv:
            plt.scatter(element[0], element[1], c = color[ci], marker='x')
        else:
            plt.scatter(element[0], element[1], c = color[ci])
            
    plt.savefig('svm_poly.png')
    
    plt.show()
    
        

