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
    data = read_csv('X_train.csv')
    data = np.array(data).astype(np.float)
    dataT = np.transpose(data)
    cov_mat = np.cov(dataT)
    #print(cov_mat)
    #print(cov_mat.tolist())

    evalue, evector = np.linalg.eig(cov_mat)
    w = []
    ind = np.argmax(evalue)
    print(max(evalue), np.argmax(evalue))
    
    #print(evector[np.argmax(evalue)])
    w.append(evector[:,ind])
    evalue = np.delete(evalue,ind)
    evector = np.delete(evector, ind, axis =1)
    ind = np.argmax(evalue)
    print(max(evalue), ind)
    w.append(evector[:,ind])
    w = np.array(w)     #[2, 784]
    print(w.T.shape)
    new_data = np.matmul(data, w.T)
    print(new_data.real)
    for i, element in enumerate(new_data.real):
        ci = int(i/1000)
        plt.scatter(element[0], element[1], c = color[ci])
    plt.show()
    
    
        

