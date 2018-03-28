import sys

def factorial(n):
    result = 1
    if n == 0:
        return 1
    for i in range(1, n+1):
        result *= i
    
    return result


if __name__ == '__main__':
    fp = open(sys.argv[1], 'r')
    initial_a = int(sys.argv[2])
    initial_b = int(sys.argv[3])
    
    a = initial_a
    b = initial_b
    for line in iter(fp):
        N = 0
        m = 0
        for c in line:
            if c == '\n':
                break
            if int(c) == 1:
                m += 1
                N += 1
            else:
                N += 1
        likelihood = m/float(N)
        print('N:%d, m:%d, likelihood = %f' %(N, m, likelihood))
        print('prior: a=%d, b=%d' %(a,b))
        print('posterior: a=%d, b=%d' %(a+m, N-m+b))
        a = a+m
        b = b+N-m
        print('')    
            
