def lr_decompose(M):
    L = M
    R = M
    return L, R



M = [ [0 for x in range(5)] for y in range(3)]
l, r = lr_decompose(M)
print(l)
print(r)
print(len(l))
