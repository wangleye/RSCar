import numpy as np
import matplotlib.pyplot as plt
import os
import operator
import math
import mapSim as mapSim
import scipy.stats as stats
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve


def loadFileAsFeatureDefaultLabel(datapath,label):
    featurearr = []
    labelarr = []
    with open(datapath) as featfile:
        for lines in featfile:
            tuples = lines.split(",")
            thisvin = tuples[0]
            thisfeature = map(lambda x:float(x),tuples[1:])
            featurearr.append(thisfeature)
            labelarr.append(label)
    return {'feature':featurearr,'label':labelarr}

def loadFileWithLabel(datapath,labelpath):
    data = {}
    featurearr = []
    labelarr = []
    with open(datapath) as featfile:
        for lines in featfile:
            tuples = lines.split(",")
            thisvin = tuples[0]
            thisfeature = map(lambda x:float(x),tuples[1:])
            data[thisvin] = {'feature':thisfeature,'label':0}
    with open(labelpath) as labelfile:
        for lines in labelfile:
            tuples = lines.split(",")
            vin = tuples[0]
            val = 0 if(tuples[1][0]=='U')else 1
            data[vin]['label'] = val
    for vin in data.keys():
        element = data[vin]
        featurearr.append(element['feature'])
        labelarr.append(element['label'])
    return {'feature':featurearr,'label':labelarr}

#taxi data -- 0
taxidata = loadFileAsFeatureDefaultLabel('../data/sjtu/oneclass/taxifeatureout.txt',0)
#bus data -- 1
busdata = loadFileAsFeatureDefaultLabel('../data/sjtu/oneclass/busfeatureout.txt',1)
tr_feature = np.asarray(taxidata['feature']+busdata['feature'])
tr_label = np.asarray(taxidata['label']+busdata['label'])
#tstset
tstset = loadFileWithLabel('../data/sjtu/oneclass/200labeledfeature.txt','../data/label.csv')
tst_feature = np.asarray(tstset['feature'])
tst_label = np.asarray(tstset['label'])

featmean = np.mean(tr_feature,axis = 0)
featstd = np.std(tr_feature,axis = 0)
tr_feature -= featmean
tr_feature /=featstd

tst_feature -= featmean
tst_feature /= featstd

model = RandomForestClassifier(n_estimators=20,criterion='entropy')
#model=SVC(probability=True,kernel='rbf')
model.fit(tr_feature,tr_label)

tr_accuracy = np.mean(model.predict(tr_feature)==tr_label)
tst_res = model.predict(tst_feature)==tst_label
tst_accuracy = np.mean(tst_res)
print tst_res
tst_pred = model.predict_proba(tst_feature)
proba = map(lambda x:max(x),tst_pred)

tst_log = []
for each in zip(tst_res,proba,tst_label):
    tst_log.append({'p':each[1],'acc':each[0],'label':each[2]})
records = sorted(tst_log,key = operator.itemgetter('p'),reverse=True)
for i in range(1,len(records)):
    r = map(lambda x:x['acc'],records[0:i])
    lr = map(lambda x:x['label'],records[0:i])
    percent = float(sum(r))/i
    records[i-1]['percent'] = percent
    records[i-1]['labelratio'] = float(sum(lr))/i
print records

distribution = {}
for r in records:
    try:
        if(r['p'] not in distribution.keys()):
            distribution[r['p']] = 1
        distribution[r['p']] = min(distribution[r['p']],r['percent'])
    except Exception,e:
        e
p = []
k = []
for keys in sorted(distribution.keys(),reverse=True):
    k.append(keys)
    p.append(distribution[keys])

print k
print p

plt.plot(k,p)
plt.xlabel("prediction confidence")
plt.ylabel("percentage of right prediction")
plt.show()
print " Train Accuracy:%f, Test Accuracy:%f" % (tr_accuracy,tst_accuracy)
