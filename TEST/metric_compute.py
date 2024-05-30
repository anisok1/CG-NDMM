from sklearn import metrics
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

a = torch.tensor([[3],
        [2],
        [3],
        [2],
        [3],
        [3]])
b = torch.tensor([[1],
        [0],
        [0],
        [0],
        [0],
        [0]])
c = a.numpy()
d = b.numpy()
a = np.append(c, d)
b = np.append(d, d)
mcm = metrics.multilabel_confusion_matrix(a, b)
m = metrics.confusion_matrix(a, b)
conf_matrix = pd.DataFrame(m, index=['cover_nature','cover_gen', 'steg_nature',"steg_gen"], columns=['cover_nature','stgo_nature','cover_gen', "stgo_gen"])
def ACC(Y_test,Y_pred,n):           
    con_mat = metrics.confusion_matrix(Y_test,Y_pred)   
    number = np.sum(con_mat[:,:]) 
    tp = 0
    for i in range(n):          
        tp += con_mat[i][i]                   
    acc = tp / number                   
    return acc

def extract(Y_test,Y_pred):
    con_mat = metrics.multilabel_confusion_matrix(Y_test,Y_pred) 
    TP = []
    FP = []
    FN = []
    TN = []
    for i in range(len(con_mat)):
            MAT = con_mat[i]   
            TP.append(MAT[0][0])        
            FP.append(MAT[1][0])
            FN.append(MAT[0][1])
            TN.append(MAT[1][1])
    return TP, FP, FN, TN

def PreAndRec(TP, FP, FN, TN):
    sump = 0
    sumc = 0
    for i in range(4):
        pre = TP[i]/ (TP[i] + FP[i])
        recall = TP[i]/ (TP[i] + FN[i])
        sump += pre
        sumc += recall
    pre = sump/ 4
    recall = sumc/ 4
    return pre, recall 

acc = ACC(a, b, 4)
TP, FP, FN, TN = extract(a, b)
pre, recall = PreAndRec(TP, FP, FN, TN)
con_mat = metrics.multilabel_confusion_matrix(a,b) 
con = metrics.confusion_matrix(a,b) 
ccc = precision_score(a, b, average='macro')
ddd = recall_score(a, b, average='macro')
eee = accuracy_score(a, b)
print(con)
print(a, b)
print(ccc, pre)
print(ddd, recall)
print(eee, acc)
