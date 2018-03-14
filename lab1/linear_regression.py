from __future__ import print_function
import sys
import show
A = []
X = []
Y = []
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

def linear_reg(A, b, bases, la):
    li = M_mul_scalar(eye(bases) ,la)
    t1 = inverse(M_add(M_mul(transpose(A), A), li))
    X = M_mul(M_mul(t1, transpose(A)), b)

    return X
   
def Hessian(x):
    return M_mul_scalar(M_mul(transpose(A), A) ,2)

def gradient(x):
    At = transpose(A)
    t1 = M_mul_scalar(M_mul(M_mul(At, A), x), 2)
    t2 = M_mul_scalar(M_mul(At, transpose([Y])), -2)
    return M_add(t1, t2)

def newton(A, b, bases):
    x0 = []
    for i in range(bases):
        x0.append([0])
    #print(x0)
    while True:
        d = M_mul(inverse(Hessian(x0)), gradient(x0))
        x = M_add(x0, M_mul_scalar(d, -1))    
        error_M_old = M_add(M_mul(A, x0), M_mul_scalar(transpose([Y]), -1))
        error_old = M_mul(transpose(error_M_old), error_M_old)

        error_M_new = M_add(M_mul(A, x), M_mul_scalar(transpose([Y]), -1))
        error_new = M_mul(transpose(error_M_new), error_M_new)
        if (error_old[0][0] - error_new[0][0]) <=1:
            break
        else:
            x0 = x 
    return x
if __name__ == '__main__':
    
    fp = open(sys.argv[1], 'r')
    bases = int(sys.argv[2])
    la = int(sys.argv[3])
    for line in iter(fp):
        l = line.strip().split(',')
        X.append(float(l[0]))
        Y.append(float(l[1]))
    #X = transpose([X])
    #Y = transpose([Y])
    
    for x in X:
        row = [1]
        for i in range(bases-1):
            row = [row[0]*x] + row
        A.append(row)
    
    ans = linear_reg(A, transpose([Y]), bases, la)
    error_M = M_add(M_mul(A, ans), M_mul_scalar(transpose([Y]), -1))
    error = M_mul(transpose(error_M), error_M)
    
    ans_newton = newton(A, transpose([Y]), bases)
    error_M_n = M_add(M_mul(A, ans_newton), M_mul_scalar(transpose([Y]), -1))
    error_n = M_mul(transpose(error_M_n), error_M_n)
    print("for LSE:")
    for i in range(bases-1):
        print('%+.2fx^%d ' %(ans[i][0], bases-i-1) , end='')
    print('%+.2f' %(ans[bases-1][0]))
    print("error: %03f" %(error[0][0]))
    print(" ")

    print("For Newton method:")
    for i in range(bases-1):
        print('%+.2fx^%d ' %(ans_newton[i][0], bases-i-1) , end='')
    print('%+.2f' %(ans_newton[bases-1][0]))
    print("error: %03f" %(error_n[0][0]))
    l1 = []
    for i in range(bases):
        l1.append(ans[bases-i-1][0])
    l2 = []
    for i in range(bases):
        l2.append(ans_newton[bases-i-1][0])
    show.show(X, Y, l1, l2)
