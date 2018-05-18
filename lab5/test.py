y = []
x = []
for line in open('test_libsvm_format'):
    line = line.split(None, 1)
    if len(line) == 1: line += ['']
    label, feature = line
    print(label)
    print(feature)
    xi = {}
    for e in feature.split():
        ind, val = e.split(":")
        xi[int(ind)] = float(val)
    x += [xi]
    print(x)
    break
