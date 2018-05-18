import sys





data = open(sys.argv[1], 'r')
label = open(sys.argv[2], 'r')
wf = open(sys.argv[3], 'w')

l = label.readline()
while l:
    d = data.readline()
    d_split = d.split(' ', 1)
    #print(l[:-1])
    #print(d_split[1])
    wf.write(l[:-1] + ' '+d_split[1])
    l = label.readline()
data.close()
label.close()
wf.close()


