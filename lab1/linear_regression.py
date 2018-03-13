import sys
def eye(n):
    M = [ [1 if (i==j) else 0  for i in range(n)] for j in range(n)]
    return M

def M_add(A, B):
    C = A[:]
    for i in range(len(A)):
        C[i] = map(lambda x,y: x+y, A[i], B[i])
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
    U[0] = map(lambda x: x/float(A[0][0]), A[0])

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
        M[i] = map(lambda x: x*s, M[i])
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
    
if __name__ == '__main__':
    A = [[1,2,3], [4,5,6] , [7,8,9]]
    B = eye(3)
    C = [[1,4,7,5], [2,3,5,3], [4,5,8,0.2],[1,0.3,-0.3,4]]
    D = [[24,3,45], [-2,7,-10], [8,21,24]]
    E = [[3,-0.1,-0.2], [0.1,7,-0.3],[0.3, -0.2,10]]
    M = M_add(A,B)
    I = [[3,18,24], [-2,-7,-36],[1,9,-2]]
    G = [[1,2,3],[4,5,6]]
    H = [[-2,35],[32,4],[0.5,1.25]]
    K = transpose(H)
    
    X = []
    Y = []
    fp = open(sys.argv[1], 'r')
    bases = int(sys.argv[2])
    la = int(sys.argv[3])
    for line in iter(fp):
        l = line.strip().split(',')
        X.append(float(l[0]))
        Y.append(float(l[1]))
    #X = transpose([X])
    #Y = transpose([Y])
    A = []
    for x in X:
        row = [1]
        for i in range(bases-1):
            row = [row[0]*x] + row
        A.append(row)
    
    ans = linear_reg(A, transpose([Y]), bases, la)
    error_M = M_add(M_mul(A, ans), M_mul_scalar(transpose([Y]), -1))
    error = M_mul(transpose(error_M), error_M)
        
    #print(X)
    #print(Y)
    #print(A)
    print(ans)
    print(error)
