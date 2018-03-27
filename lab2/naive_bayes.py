import sys
import math

bin_num = 8
def discrete(image_fp, label_fp, np, n):
    
    table = [[[1 for i in range(10)] for j in range(bin_num)] for k in range(np)]  #table[784][32][10]
    num_per_digit = [0 for i in range(10)]
    for i in range(n):
        label = int.from_bytes(label_fp.read(1), byteorder='big')
        num_per_digit[label] += 1
        for j in range(np):
            value = int(int.from_bytes(image_fp.read(1), byteorder='big') / (256.0/bin_num)) # 256 into 32 bin
            table[j][value][label]+=1
    

    return table, num_per_digit

def discrete_predict(table, num_per_digit, np, n, fp):
    
    posterior = [0.0 for i in range(10)]

    image = [None]*np
    for i in range(np):
        image[i] = int(int.from_bytes(fp.read(1), byteorder='big')/(256.0/bin_num))

    for i in range(10):
        for j in range(np):
            value = image[j]
            pj = 0
            if table[j][value][i] == 0 :
                pj = 1 / float(num_per_digit[i])
            else :
                pj = table[j][value][i] / float(num_per_digit[i])
            posterior[i] += math.log(pj)
        posterior[i] += math.log(num_per_digit[i] / float(n))       
    
    return posterior

def continuous(image_fp, label_fp, np,n):

    table = [[[0 for i in range(2)] for j in range(10)] for k in range(np)]  #table[784][10][2]  0:x 1:x^2
    num_per_digit = [0 for i in range(10)]
    for i in range(n):
        label = int.from_bytes(label_fp.read(1), byteorder='big')
        num_per_digit[label] += 1
        for j in range(np):
            value = int.from_bytes(image_fp.read(1), byteorder='big')
            table[j][label][0] += value
            table[j][label][1] += value*value
            
    return table, num_per_digit

def continuous_predict(table, num_per_digit, np, n, fp):

    posterior = [1.0 for i in range(10)]

    image = [None]*np
    for i in range(np): 
        image[i] = int.from_bytes(fp.read(1), byteorder='big')

    for i in range(10):
        for j in range(np):
            value = image[j]
            mean = table[j][i][0] / float(num_per_digit[i])
            var = (table[j][i][1] / float(num_per_digit[i])) - mean**2
            if var == 0 :
                var = 0.0001
            value = (value-mean) / float(var**0.5)
            pj = math.exp((-value**2/(2.0)) / math.sqrt(2.0*math.pi))
            if pj<0.0001:
                pj = 0.0001
            posterior[i] += math.log(pj)
        posterior[i] += math.log(num_per_digit[i] / float(n))

    return posterior
            

if __name__ == '__main__':
    
    image_fp = open(sys.argv[1], 'rb')
    label_fp = open(sys.argv[2], 'rb')
    test_im = open(sys.argv[3], 'rb')
    test_lb = open(sys.argv[4], 'rb')
    mode = int(sys.argv[5])

    magic_number = int.from_bytes(image_fp.read(4), byteorder='big')
    n = int.from_bytes(image_fp.read(4), byteorder='big')
    image_row = int.from_bytes(image_fp.read(4), byteorder='big')
    image_col = int.from_bytes(image_fp.read(4), byteorder='big')
    
    num_of_pixels = image_row * image_col
    label_fp.seek(8)
    if mode == 0:
        t, num_per_digit = discrete(image_fp, label_fp, num_of_pixels, n)
    else :
        t, num_per_digit = continuous(image_fp, label_fp, num_of_pixels, n)
    test_im.read(4)
    test_n = int.from_bytes(test_im.read(4), byteorder='big')
    test_im.read(8)
    test_lb.read(8)
    match = 0
    
    for i in range(test_n):
        if mode == 0:
            p = discrete_predict(t, num_per_digit, num_of_pixels, n, test_im)
        else :
            p = continuous_predict(t, num_per_digit, num_of_pixels, n, test_im)
        predict = p.index(max(p))
        real_y = int.from_bytes(test_lb.read(1), byteorder='big')
        #print(predict, real_yi, p)
        if predict == real_y :
            match += 1
    #print(p)
    print(match / float(test_n))
    
    #print(t)


