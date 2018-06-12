from svm import *
from svmutil import *
import sys




if __name__ == '__main__':
    y, x = svm_read_problem('train_precompute_data')
    ty, tx = svm_read_problem('test_precompute_data')
    prob = svm_problem(y, x, isKernel = True)
    param = svm_parameter('-t 4 -c 4 ')
    param_poly = svm_parameter('-t 1')
    param_linear = svm_parameter('-t 0 ')
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(ty, tx, model)
    #print(p_acc)
    
