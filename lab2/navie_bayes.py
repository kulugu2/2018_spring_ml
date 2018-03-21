import sys

def discrete(image_fp, label_fp, np, n):
    
    table = [[[0 for i in range(10)] for j in range(32)] for k in range(np)]
    num_per_digit = [0 for i in range(10)]
    for i in range(n):
        label = int.from_bytes(label_fp.read(1), byteorder='big')
        num_per_digit[label] += 1
        for j in range(np):
            value = int(int.from_bytes(image_fp.read(1), byteorder='big') / 8) # 256 into 32 bin
            table[j][value][label]+=1
    

    return table, num_per_digit


if __name__ == '__main__':
    
    image_fp = open(sys.argv[1], 'rb')
    label_fp = open(sys.argv[2], 'rb')

    magic_number = int.from_bytes(image_fp.read(4), byteorder='big')
    n = int.from_bytes(image_fp.read(4), byteorder='big')
    image_row = int.from_bytes(image_fp.read(4), byteorder='big')
    image_col = int.from_bytes(image_fp.read(4), byteorder='big')
    
    num_of_pixels = image_row * image_col
    label_fp.seek(8)
    t, num_per_digit = discrete(image_fp, label_fp, num_of_pixels, n)
    print(t)

