import dataLoader as loader
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve

consts = {
    'boundary':0.55,
    'drawROC':False,
    'normpath':"./model/norm",
    'svmpath':"./model/svm"
}
f = loader.featureFile()
vinList= map(lambda x:x['vin'],f)
featureList = np.asarray(map(lambda x:x['feature'],f))
labelList = np.asarray(map(lambda x:x['label'],f))

featmean = np.mean(featureList,axis = 0)
featstd = np.std(featureList,axis = 0)
featureList -= featmean
featureList /=featstd

kf=KFold(n=len(vinList),n_folds=5,shuffle=True)
cv=0
tstacc = []
tracc = []
if(consts['drawROC']):
    plt.figure()
    th = []

for tr,tst in kf:
    tr_features=featureList[tr,:]
    tr_target=labelList[tr]
    tst_features=featureList[tst,:]
    tst_target=labelList[tst]

    model=SVC(probability=True,kernel='rbf')
    model.fit(tr_features,tr_target)

    tr_accuracy = np.mean(model.predict(tr_features)==tr_target)
    tst_accuracy = np.mean(model.predict(tst_features)==tst_target)
    tst_pred = model.predict_proba(tst_features)
    if(consts['drawROC']):
        result_probability = map(lambda x:x[1],tst_pred)
        label_roc = map(lambda x:int(x),tst_target)
        fpr, tpr, thresholds = roc_curve(label_roc, result_probability)
        a = auc(fpr,tpr)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % a)
        metrics = np.asarray(1-fpr+tpr)
        threshold = np.asarray(thresholds)
        optimal_threashold = np.sum(metrics*threshold)/np.sum(metrics)

    tracc.append(tr_accuracy)
    tstacc.append(tst_accuracy)

    print "%d Fold Train Accuracy:%f, Test Accuracy:%f" % (cv,tr_accuracy,tst_accuracy)
    th.append(optimal_threashold)
    cv+=1

if(consts['drawROC']):
    print "training accuracy: %f" % (np.mean(tracc))
    print "testing accuracy: %f" % (np.mean(tstacc))
    op_th = np.sum(np.array(th)*tstacc)/float(np.sum(tstacc))
    print op_th
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()