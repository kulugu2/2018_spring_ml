import sys


class MNIST(object):
    def __init__(self, path):
        self.path = path

        self.train_im_fname = 'train-images.idx3-ubyte'
        self.train_lb_fname = 'train-labels-idx1-ubyte'
        self.test_im_fname = 't10k-images-idx3-ubyte'
        self.test_lb_fname = 't10k-labels-idx1-ubyte'

        self.train_im = []
        self.train_lb = []
        self.test_im = []
        self.test_lb = []

    def load(self):
        train_im_fp = open(self.path+self.train_im_fname, 'rb')
        train_lb_fp = open(self.path+self.train_lb_fname, 'rb')
        test_im_fp = open(self.path+self.test_im_fname, 'rb')
        test_lb_fp = open(self.path+self.test_lb_fname, 'rb')
        
        magic = int.from_bytes(train_im_fp.read(4), byteorder='big')
        n = int.from_bytes(train_im_fp.read(4), byteorder='big')
        row = int.from_bytes(train_im_fp.read(4), byteorder='big')
        col = int.from_bytes(train_im_fp.read(4), byteorder='big')
        print(n, col, row) 
        for i in range(n):
            #pixel = []
            #for j in range(row*col):
            #pixel.append(int.from_bytes(train_im_fp.read(1), byteorder='big'))
            a = train_im_fp.read(784)
            
            #self.train_im.append(pixel)
        b = a[0]
        print(b) 
        train_lb_fp.seek(8)
        for i in range(n):
            train_lb_fp.read(1)
            #self.train_lb.append([int.from_bytes(train_lb_fp.read(1), byteorder='big')])
# test set

        magic = int.from_bytes(test_im_fp.read(4), byteorder='big')
        n = int.from_bytes(test_im_fp.read(4), byteorder='big')
        row = int.from_bytes(test_im_fp.read(4), byteorder='big')
        col = int.from_bytes(test_im_fp.read(4), byteorder='big')

        for i in range(n):
            pixel = []
            for j in range(row*col):
                pixel.append(int.from_bytes(test_im_fp.read(1), byteorder='big'))
            #self.test_im.append(pixel)

        test_lb_fp.seek(8)
        #for i in range(n):
            #self.test_lb.append([int.from_bytes(test_lb_fp.read(1), byteorder='big')])

