import sys
import math
from mnist import MNIST
mnist_repo = '../lab2/'





if __name__ == '__main__':
    '''
    train_im = open(mnist_repo+'train-images.idx3-ubyte', 'rb')
    train_lb = open(mnist_repo+'train-labels.idx1-ubyte', 'rb')
    test_im = open(mnist_repo+'t10k-images.idx3-ubyte', 'rb')
    test_lb = open(mnist_repo+'t10k-labels.idx1-ubyte', 'rb')
    '''
    #train_im = open('../lab2/train-images.idx3-ubyte', 'rb')
    #print(train_im.read(4))
    
    mnist = MNIST(mnist_repo)
    mnist.load()

    print(mnist.train_im)

