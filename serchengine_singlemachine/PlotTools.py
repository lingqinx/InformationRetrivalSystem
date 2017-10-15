# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def precise_recall(model_result,true_result):
    #input type = numpy.darray
    model_set = set(model_result.tolist())
    true_set = set(true_result.tolist())
    uni_set = model_set & true_set
    if len(uni_set) == 0:
        return 0,0
    precise = 1.0*len(uni_set)/len(model_set)
    recall = 1.0*len(uni_set)/len(true_set)
    return precise,recall
    
def precise_recall_plot(precise,recall,name,color):
    plt.figure(figsize=(8,8),dpi=80)
    for i in xrange(len(precise)):
        plt.plot(recall[i],precise[i],'%s-' % color[i],linewidth=1.5,label=name[i])
    plt.ylabel(r'$Precise$')
    plt.xlabel(r'$Recall$')

    plt.legend(loc='upper right', frameon=True)
    #plt.xlim(0,1)
    #plt.ylim(0,1)
    plt.show()

    
if __name__ == '__main__':
    a = np.linspace(0,10,256,endpoint=True)
    b = a**2-a+1
    precise_recall_plot(a,b)

