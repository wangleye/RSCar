import numpy as np
import rawdataloader as loader
import cnn as cnn
from sklearn.cross_validation import KFold


cnndata = loader.mapFile()
cnnclassify = cnn.classify(cnndata)
cnnres = cnnclassify['detail']
record = []
for each in cnnres.keys():
    r = cnnres[each]
    record.append({'vin':each,'confidence':max(r['predsum']),'result':r['predres'],'label':r['label']})
sortedrecord = sorted(record, key=lambda k: k['confidence'],reverse=True)
print sortedrecord
cnnacc = cnnclassify['accuracy']
print "standalone classifier accuracy, cnn -- %f" % (cnnacc)
