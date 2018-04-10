import sys
import random
import math
import matplotlib.pyplot as plt
import linear_model_data_generator as gen

def eye(n):
    M = [ [1 if (i==j) else 0  for i in range(n)] for j in range(n)]
    return M

def M_add(A, B):
    C = A[:]
    for i in range(len(A)):
        C[i] = list(map(lambda x,y: x+y, A[i], B[i]))
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


if __name__ == '__main__':
    b = float(sys.argv[1])
    basis = int(sys.argv[2])
    a = float(sys.argv[3])
    w = []
    for i in range(basis):
        w.append(float(sys.argv[4+i]))
    
    m = [[0],[0],[0]]
    precision = M_mul_scalar(eye(basis), b)
    while True:
        x, y = gen.linear_gen(w,a)
        print(x, y)
        design = []
        for i in range(basis):
            design.append(x**i)
        old_precision = precision
        precision = M_add(M_mul_scalar(M_mul(transpose([design]), [design]), 1.0/a), M_mul_scalar(eye(basis),b) )   #aXtX + bI
        m = M_mul(inverse(precision), M_add(M_mul_scalar(transpose([design]), (1.0/a)*y), M_mul(old_precision, m)))
        print(transpose(m))
        input()



