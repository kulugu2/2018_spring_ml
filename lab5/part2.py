from svm import *
from svmutil import *
import sys




if __name__ == '__main__':
    y, x = svm_read_problem('train_libsvm_format')
    ty, tx = svm_read_problem('test_libsvm_format')
    prob = svm_problem(y, x)
    param = svm_parameter('-t 2 -c 4 -b 0')
    param_poly = svm_parameter('-t 1')
    param_linear = svm_parameter('-t 0 ')
    model = svm_train(prob, param_linear)
    svm_save_model('linear_model', model)
    sv = model.get_sv_indices()
    print(len(sv))
    print(sv)
    p_label, p_acc, p_val = svm_predict(ty, tx, model)
    #print(p_acc)
    
