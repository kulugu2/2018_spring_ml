import sys
import random
import math
import matplotlib.pyplot as plt
from univariate_gaussian_data_generator import univariate_gen as gen
import numpy as np

def eye(n):
    M = [ [1 if (i==j) else 0  for i in range(n)] for j in range(n)]
    return M

def M_add(A, B):
    C = A[:]
    for i in range(len(A)):
        C[i] = list(map(lambda x,y: x+y, A[i], B[i]))
    return C

def M_sub(A, B):
    C = A[:]
    for i in range(len(A)):
        C[i] = list(map(lambda x,y: x-y, A[i], B[i]))
    return C

def M_mul(A, B):
    C = [ [0 for i in range(len(B[0]))] for j in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(A[0])):
                C[i][j] += A[i][k] * B[k][j]
    return C

def LU_decompose(A):
    n = len(A)
    L = [[A[j][0] if i==0 else 0 for i in range(n)] for j in range(n)]
    U = eye(n)
    U[0] = list(map(lambda x: x/float(A[0][0]), A[0]))

    for i in range(1,n):
        for j in range(i,n):
            L[j][i] = A[j][i]
            for k in range(i):
                L[j][i] -= L[j][k]*U[k][i]

        for j in range(i+1,n):
            U[i][j] = A[i][j]
            for k in range(i):
                U[i][j] -= L[i][k]*U[k][j]
            U[i][j] = U[i][j]/float(L[i][i])

                
    return L, U

def  M_mul_scalar(M, s):
    n = len(M)
    for i in range(n):
        M[i] = list(map(lambda x: x*s, M[i]))
    return M

def inverse(M):
    #print(M[:])
    L, U = LU_decompose(M)
    #print(L[:])
    n = len(L)

    """ L inverse  """
    LI = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        LI[i][i] = 1/float(L[i][i])
        for j in range(i+1, n):
            for k in range(j):
                LI[j][i] -= L[j][k]*LI[k][i]
            LI[j][i] = LI[j][i]/float(L[j][j])
    #print(LI[:])
    
    """ U inverse """
    #print("U")
    #print(U[:])
    UI = [[0 for i in range(n)] for j in range(n)]
    for i in range(n-1, -1, -1):
        UI[i][i] = 1/float(U[i][i])
        for j in range(i-1, -1, -1):
            for k in range(n-1, j, -1):
                UI[j][i] -= U[j][k]*UI[k][i]
            UI[j][i] = UI[j][i]/float(U[j][j])

    #print(UI[:])

    MI = M_mul(UI, LI)
    return MI

def transpose(M):
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]

def npoint(m1, v1, m2, v2, n):
    D = []
    for i in range(n):
        x = gen(m1, v1)
        y = gen(m2, v2)
        D.append([x, y])
    return D

def sigmoid(m):
    for i in range(len(m)):
        for j in range(len(m[i])):
            #print(m[i][j])
            if -m[i][j] > 700:
                m[i][j] = 0
            else:
                m[i][j] = 1.0 / (1 + math.exp(-m[i][j]))
    return m

def M_log(m):
    for i in range(len(m)):
        for j in range(len(m[i])):
            if m[i][j] < sys.float_info.min:
                m[i][j] = sys.float_info.min
            m[i][j] = math.log(m[i][j])
    return m

def one(n):
    m = []
    for i in range(n):
        m.append([1.0])
    return m 

def cost(X, W, Y):
    c = M_mul(transpose(Y), M_log(sigmoid(M_mul(X, W))))[0][0]
    c += M_mul(transpose(M_sub(one(len(Y)), Y)), M_log(M_sub(one(len(Y)), sigmoid(M_mul(X,W)))))[0][0]
    return c

def gd(X, W, Y, a, n):
    i = 0 
    old_cost = sys.float_info.min
    while True:
        gradient = M_mul(transpose(X), M_sub(sigmoid(M_mul(X,W)), Y))
        old_w = W
        W = M_sub(W, M_mul_scalar(gradient, a*(n / float(n+i))))
        c = cost(X,W,Y)
        print("cost: %.3f" %( c))
        print(W)
        confusion(X,W,Y)
        i += 1
        b = 1
        if c - old_cost > 10:
            old_cost = c
            b = 0
        for j in range(len(W)):
            if(abs((W[j][0] - old_w[j][0]) / W[j][0]) > 0.01):
                b = 0
                break
        old_cost = c
        if b == 1:
            print(i)
            return W

def confusion(X,W,Y):
    p = sigmoid(M_mul(X,W))
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(p)):
        if p[i][0] > 0.5:
            if Y[i][0] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if Y[i][0] == 1:
                fn += 1
            else:
                tn += 1
    print('sensitivity: %.3f' %(tp/float(tp+fn)))
    print('specificity: %.3f' %(fp/float(fp+tn)))
    return

def hessian(X, W):
    D = [[0 for i in range(len(X))] for j in range(len(X))]
    for i in range(len(X)):
        zi = M_mul([X[i]], W)[0][0]
        if (-zi) > 700:
            zi = 0
        else:
            zi = 1.0 / (1.0 + math.exp(-zi))
        D[i][i] = zi
    h = M_mul(M_mul(transpose(X), D), X)
    return h

def newton(X, W, Y, n):
    w = W
    i = 0
    old_cost = sys.float_info.min
    while True:
        old_w = w
        try:
            w = M_sub(w, M_mul(inverse(hessian(X,w)), M_mul(transpose(X), M_sub(sigmoid(M_mul(X,w)), Y))))
        except ZeroDivisionError:
            print("hessian can not inverse, use gradient descent")
            
            gradient = M_mul(transpose(X), M_sub(sigmoid(M_mul(X,w)), Y))
            w = M_sub(w, M_mul_scalar(gradient, 1*(n / float(n+i))))
        
        c = cost(X,W,Y)
        print("cost: %.3f" %( c))
        print(w)
        confusion(X,w,Y)
        i += 1
        b = 1
        if c - old_cost > 10:
            old_cost = c
            b = 0
        for j in range(len(w)):
            if(abs((w[j][0] - old_w[j][0]) / w[j][0]) > 0.01):
                b = 0
                break
        old_cost = c
        if b == 1:
            print(i)
            return w
    
if __name__ == '__main__':
    #usage python3 logistic_reg.py n mx1 vx1 my1 vy1 mx2 vx2 my1 vy2
    n = int(sys.argv[1])
    mx1 = float(sys.argv[2])
    vx1 = float(sys.argv[3])
    my1 = float(sys.argv[4])
    vy1 = float(sys.argv[5])
    mx2 = float(sys.argv[6])
    vx2 = float(sys.argv[7])
    my2 = float(sys.argv[8])
    vy2 = float(sys.argv[9])
    
    D1 = npoint(mx1, vx1, my1, vy1, n)
    D2 = npoint(mx2, vx2, my2, vy2, n)
    D1.extend(D2)

    X = D1
    for row in X:
        row.append(1.0)
    #print(X)
    Y = []
    W = [[2.0], [0.5],[1.0]]
    
    #print(sigmoid(M_mul(X,W)))
    for i in range(n):
        Y.append([0.0])
    for i in range(n):
        Y.append([1.0])
    #X = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]
    #Y = [[0.0],[1.0],[0.0]]
    #c = cost(X,W,Y)
    #print(c)
    #print("cost: %.3f" %( cost(X,W,Y)))
    #print(W)
    #gd(X,W,Y,1, n)
    newton(X,W,Y,n)

