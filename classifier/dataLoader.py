import os
import numpy as np

consts = {
    'featurefile':"../feature/feature_extracted.txt",
    'mapfile':"../feature/map/full/",
    'labelfile':"../data/label.csv",
}
def labelFile():
    data = {}
    with open(consts['labelfile'],'r') as f:
        for line in f:
            tuples = line.split(",")
            key = tuples[0]
            val = 0 if(tuples[1][0]=='U')else 1
            data[key]= val
    return data

def featureFile():
    data = []
    labeldict = labelFile()
    with open(consts['featurefile'],'r') as f:
        for line in f:
            tuples = line.split(":")
            vin = tuples[0]
            feat = tuples[1].rstrip().split(",")
            features = map(lambda x:float(x),feat)
            label = labeldict[vin]
            data.append({'vin':vin,'feature':features,'label':label})
    return np.asarray(data)

def mapFile():
    data = {}
    labeldict = labelFile()
    datfiles = os.listdir(consts['mapfile'])
    for f in datfiles:
        tuples = f.split(".")
        vin = tuples[0]
        date = tuples[1]
        a = [0,0]
        label = labeldict[vin]
        a[label] = 1
        label_one_hot = np.asarray(a).reshape(1,2)
        mat = np.loadtxt(consts['mapfile']+f).reshape(1,576)
        if vin not in data.keys():
            data[vin] = {'feature':[],'label':label_one_hot}
        data[vin]['feature'].append(mat)
    return data
