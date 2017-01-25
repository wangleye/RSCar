import numpy as np
import dataLoader as loader
import svm as svm
import cnn as cnn
from sklearn.cross_validation import KFold

const={
    "decision_boundary":1.1,
}

svmdata = loader.featureFile()
cnndata = loader.mapFile()
labeldata = loader.labelFile()
vinlist = np.array(labeldata.keys())
kf=KFold(n=len(vinlist),n_folds=5,shuffle=True)
cv = 0
for tr,tst in kf:
    cv+=1
    print "cross validation fold %d"%(cv)
    trvin = vinlist[tr]
    tstvin = vinlist[tst]

    svmtrain = filter(lambda x:x['vin'] in trvin,svmdata)
    svmtest = filter(lambda x:x['vin'] in tstvin,svmdata)
    cnntrain = {}
    cnntest = {}
    for k in cnndata.keys():
        if(k in trvin):
            cnntrain[k] = cnndata[k]
        if(k in tstvin):
            cnntest[k] = cnndata[k]
    svm.train(svmtrain)
    cnn.train(cnntrain)

    svmclassify = svm.classify(svmtest)
    svmres = svmclassify['detail']
    svmacc = svmclassify['accuracy']
    cnnclassify = cnn.classify(cnntest)
    cnnres = cnnclassify['detail']
    cnnacc = cnnclassify['accuracy']
    print "standalone classifier accuracy: svm -- %f , cnn -- %f" % (svmacc,cnnacc)

    pred = {}
    for each in svmres:
        vin = each['vin']
        svm_proba = each['proba_predicted']
        cnn_proba = cnnres[vin]['predsum']
        stack_proba = (svm_proba+cnn_proba)[1]
        pred_label = 1 if(stack_proba>const['decision_boundary']) else 0
        pred[vin] = pred_label
    correct = 0
    for each in tstvin:
        if(pred[each]==labeldata[each]):
            correct+=1
    stack_accuracy = float(correct)/len(tstvin)
    print "stack classifier accuracy %f"%(stack_accuracy)