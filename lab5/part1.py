import sys
import random
import matplotlib.pyplot as plt
import math
file_name = 'moon.txt'
#file_name = 'circle.txt'
c = 2
kernel = 1
garma = 32.0
color = ['r', 'b', 'g', 'yellow']
def kf(a, b):
    if kernel == 0:      # linear kernel
        return a[0]*b[0] + a[1]*b[1]
    elif kernel == 1:      # rbf kernel
        euclidean_dist = a[0]*a[0]+a[1]*a[1] -2*(a[0]*b[0] + a[1]*b[1]) + b[0]*b[0] + b[1]*b[1]
        return math.exp(-1*garma*euclidean_dist)
if __name__ == '__main__':
    point = []
    for line in open(file_name, 'r'):
        p = line.split(',')
        point.append([float(p[0]), float(p[1])])
    n = len(point)
    label = [random.randint(0, c-1) for i in range(len(point))]
    num = [0 for i in range(c)]
    for i in range(n):
        plt.scatter(point[i][0], point[i][1], c = color[label[i]])        
    plt.show()
    for it in range(100):
        
        num = [0 for l in range(c)]
        cluster = [[] for l in range(c)]
        for i in range(n):
            num[label[i]] += 1
            cluster[label[i]].append(point[i])
        third_term = [0 for l in range(c)]
        # compute third term
        for k in range(c):
            for i in range(num[k]):
                for j in range(num[k]):
                    third_term[k] += kf(cluster[k][i], cluster[k][j])

        new_label = []
        for i in range(n): #n
            distance = [0 for k in range(c)]
            for j in range(c):
                first_term = kf(point[i], point[i])
                #print(i, point[i])
                #print(first_term)
                second_term = 0
                for k in range(num[j]):
                    second_term += kf(point[i], cluster[j][k])
                #print(second_term)
                distance[j] = first_term - 2*second_term/num[j] + third_term[j]/(num[j]**2)
            #print(distance)
            new_label.append(distance.index(min(distance)))

        label = new_label
        print(len(cluster[0]), num[0])
        print(len(cluster[1]), num[1])
        if it%10 == 0:
            for i in range(n):
                plt.scatter(point[i][0], point[i][1], c = color[label[i]])        
            plt.show()
        #print(label)


