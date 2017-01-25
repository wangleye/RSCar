import os
import numpy as np

consts = {
    'mapfile':"../data/sjtu/allmat/",
    'labelfile':"../data/sjtu/label.csv",
}
def labelFile():
    data = {}
    with open(consts['labelfile'],'r') as f:
        for line in f:
            tuples = line.split(",")
            key = tuples[0]
            val = 0 if(tuples[1][0]=='0')else 1
            data[key]= val
    return data

def mapFile():
    data = {}
    labeldict = labelFile()
    datfiles = os.listdir(consts['mapfile'])
    print len(datfiles)
    c = 0
    for f in datfiles:
        c +=1
        if(c%10000==0):
            print c
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
        mat = np.rot90(mat)
        data[vin]['feature'].append(mat)
        mat = np.rot90(mat)
        data[vin]['feature'].append(mat)
        mat = np.rot90(mat)
        data[vin]['feature'].append(mat)
    return data
