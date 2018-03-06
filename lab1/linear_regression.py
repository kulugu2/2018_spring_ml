def eye(n):
    M = [ [1 if (i==j) else 0  for i in range(n)] for j in range(n)]
    return M

def M_add(A, B):
    C = A[:]
    for i in range(len(A)):
        C[i] = map(lambda x,y: x+y, A[i], B[i])
    return C

def M_mul(A, B):
    C = [ [0 for i in range(len(A))] for j in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A)):
            for k in range(len(A)):
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


if __name__ == '__main__':
    A = [[1,2,3], [4,5,6] , [7,8,9]]
    B = eye(3)
    C = [[1,4,7,5], [2,3,5,3], [4,5,8,0.2],[1,0.3,-0.3,4]]
    D = [[24,3,45], [-2,7,-10], [8,21,24]]
    E = [[3,-0.1,-0.2], [0.1,7,-0.3],[0.3, -0.2,10]]
    M = M_add(A,B)
    L,U = LU_decompose(C)
    print(range(3,3))
#    for i in M2 :
#        print(i)
#    print(A[:])  
    print(L[:])
    print(U[:])
    print(M_mul(L,U)[:])
