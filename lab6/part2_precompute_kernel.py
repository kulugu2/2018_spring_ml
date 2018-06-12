from svm import *
from svmutil import *
import sys
import numpy as np
import matplotlib.pyplot as plt

def read_csv(fn):
    A = []
    with open(fn, 'r') as fp:
        for line in fp.readlines():
            A.append(line.strip().split(','))
    return A


color = ['r', 'g', 'b', 'y', 'c']
if __name__ == '__main__':
    y, x = svm_read_problem('../lab5/precompute_libsvm_format')
    #ty, tx = svm_read_problem('./lab5/test_precompute_data')
    prob = svm_problem(y, x, isKernel = True)
    param = svm_parameter('-t 4 -c 4 ')
    param_poly = svm_parameter('-t 1')
    param_linear = svm_parameter('-t 0 ')
    model = svm_train(prob, param)
    #p_label, p_acc, p_val = svm_predict(ty, tx, model)
    #print(p_acc)
    
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

    sv = model.get_sv_indices()
    for i, element in enumerate(new_data.real):
        ci = int(i/1000)
        if i+1 in sv:
            plt.scatter(element[0], element[1], c = color[ci], marker='x')
        else:
            plt.scatter(element[0], element[1], c = color[ci])
            
    plt.savefig('svm_precompute.png')
    plt.show()
