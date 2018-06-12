import sys
import numpy as np
import matplotlib.pyplot as plt
def read_csv(fn):
    A = []
    with open(fn, 'r') as fp:
        for line in fp.readlines():
            A.append(line.strip().split(','))
    return A

def lda(data, label, k):
    color = ['r', 'g', 'b', 'y', 'c']
    num = [0.0 for i in range(k)]
    cluster_center = np.zeros((k,len(data[0])))
    total_center = np.zeros(len(data[0]))
    for i, l in enumerate(label):
        num[l-1] += 1
        cluster_center[l-1] += data[i]
        total_center += data[i]
    for i in range(k):
        cluster_center[i] /= num[i]
    total_center /= len(data)
    
    sw = np.zeros((len(data[0]), len(data[0])))
    sb = np.zeros((len(data[0]), len(data[0])))
    
    for i in range(len(data)):
        l = label[i]
        l -= 1
        sw += np.matmul((data[i]-cluster_center[l]).reshape(1,len(data[0])).T
                , (data[i]-cluster_center[l]).reshape(1,len(data[0])))
    #print(sw[100])
    #print(sw[1])
    for i in range(k):
        sb += num[i]*np.matmul((cluster_center[i]-total_center).reshape(1, len(data[0])).T,
                (cluster_center[i]-total_center).reshape(1,  len(data[0])))
    #print(np.matmul(np.linalg.pinv(sw), sb).shape)
    evalue, evector = np.linalg.eig(np.matmul(np.linalg.pinv(sw), sb))

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
    plt.savefig('lda.png')
    plt.show()
if __name__ == '__main__':
    data = read_csv('X_train.csv')
    data = np.array(data).astype(np.float)
    label = []
    for i in range(5000):
        label.append(int(i/1000))
    lda(data, label, 5)
