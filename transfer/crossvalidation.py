import numpy as np
import dataLoader as loader
import cnn as cnn
from sklearn.cross_validation import KFold


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

    cnntrain = {}
    cnntest = {}
    for k in cnndata.keys():
        if(k in trvin):
            cnntrain[k] = cnndata[k]
        if(k in tstvin):
            cnntest[k] = cnndata[k]
    cnn.train(cnntrain)

    cnnclassify = cnn.classify(cnntest)
    cnnres = cnnclassify['detail']
    cnnacc = cnnclassify['accuracy']
    print "standalone classifier accuracy: cnn -- %f" % (cnnacc)