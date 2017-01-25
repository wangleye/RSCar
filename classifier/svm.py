import dataLoader as loader
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve
from sklearn.externals import joblib

consts = {
    'normpath':"./model/norm",
    'svmpath':"./model/svm",
    'decision_boundary':0.55,
}

'''
input train data:
[
    {
        'vin':vin,
        'feature':[],
        'label' = 1/0
    }
]
'''
def train(data):
    vinList= map(lambda x:x['vin'],data)
    featureList = np.asarray(map(lambda x:x['feature'],data))
    labelList = np.asarray(map(lambda x:x['label'],data))

    #normalization
    featmean = np.mean(featureList,axis = 0)
    featstd = np.std(featureList,axis = 0)
    featureList -= featmean
    featureList /=featstd
    content = {"mean":featmean,"std":featstd}
    joblib.dump(content,consts['normpath'])

    #train model
    model=SVC(probability=True,kernel='rbf')
    model.fit(featureList,labelList)
    joblib.dump(model, consts['svmpath'])

    #train accuracy
    tr_pred = map(lambda x:x[1]>consts['decision_boundary'],model.predict_proba(featureList))
    tr_accuracy = np.mean(tr_pred==labelList)
    return tr_accuracy

'''
input test data:
[
    {
        'vin':vin,
        'feature':[],
        'label' = 1/0 -- optional
    }
]


output detail:
[
    {
        'vin':vin,
        'feature':[],
        'label' = 1/0,
        'proba_predicted':[p0,p1],
        'label_predicted':1/0,
}
]

'''
def classify(data):
    labeled = False
    if('label' in data[0].keys()):
        labeled = True
        labelList = np.asarray(map(lambda x:x['label'],data))
    vinList= map(lambda x:x['vin'],data)
    featureList = np.asarray(map(lambda x:x['feature'],data))

    normalization = joblib.load(consts['normpath'])
    feat_mean = normalization['mean']
    feat_std = normalization['std']

    svm = joblib.load(consts['svmpath'])

    featureList -= feat_mean
    featureList /= feat_std

    predicted = np.asarray(svm.predict_proba(featureList))
    retdata = []
    correct = 0
    for each in zip(data,predicted):
        d = each[0]
        p = each[1]
        d['proba_predicted'] = p
        d['label_predicted'] = 1 if(p[1]>consts['decision_boundary']) else 0
        if(labeled):
            if(d['label_predicted']==d['label']):
                correct +=1
        retdata.append(d)
    accuracy = float(correct)/len(data) if(labeled) else 0
    return {'detail':retdata,"accuracy":accuracy}


