import sys

file_name = 'moon.txt'


if __name__ == '__main__':
    point = []
    for line in open(file_name, 'r'):
        p = line.split(',')
        point.append([float(p[0]), float(p[1])])


    print(len(point))
