import numpy as np
import matplotlib.pyplot as plt
import os
import operator
import math
import mapSim as mapSim
import scipy.stats as stats

consts = {
    'taxifeaturein':'../data/sjtu/oneclass/taxifeature.txt',
    'taxifeatureout':'../data/sjtu/oneclass/taxifeatureout.txt',
    'taximap':'../data/sjtu/oneclass/tmap/',
    'busfeaturein':'../data/sjtu/oneclass/busfeature.txt',
    'busfeatureout':'../data/sjtu/oneclass/busfeatureout.txt',
    'busmap':'../data/sjtu/oneclass/bmap/',
    'leftmax' : 121.0,
    'rightmax': 121.8,
    'topmax'  : 31.4,
    'botmax'  : 30.7,
    'grid':24,
    'passweight':0.3,
    'smoothwindow':9,
    'kmperlat':111,
    'kmperlng':95.14557,
}
def validate(feature):
    #feature = [feat_dist_mean,feat_dist_entropy,feat_s1_mean,feat_s1_entropy,feat_s2_mean,feat_s2_entropy,feat_s3_mean,feat_s3_entropy,feat_intradatesim_mean,feat_intradatesim_entropy]
    dist = 10<feature[0]<1000
    nan = (np.sum(np.isnan(feature)) == 0)
    if(dist and nan):
        return True
    else:
        return False

#taxi
taxidata = {}
with open(consts['taxifeaturein']) as featf:
    outf = open(consts['taxifeatureout'],'w')
    for lines in featf:
        t = lines.split(",")
        vin = t[0]
        date = t[1]
        distance = t[2]
        cmean = t[3]
        centropy = t[4]
        if(vin not in taxidata.keys()):
            taxidata[vin]={}
        taxidata[vin][date] = {'vin':vin,'date':date,'distance':float(distance),'cmean':float(cmean),'centropy':float(centropy)}

for vin in taxidata.keys():
    dist = []
    cmean = []
    centropy = []
    innerdatemapsim = []
    matsum = []
    intradatemapsim = []

    for date in taxidata[vin].keys():
        attr = taxidata[vin][date]
        matfull = np.loadtxt(consts['taximap']+vin+'.'+date+'.full.mat')
        mat1 = np.loadtxt(consts['taximap']+vin+'.'+date+'.part1.mat')
        mat2 = np.loadtxt(consts['taximap']+vin+'.'+date+'.part2.mat')
        mat3 = np.loadtxt(consts['taximap']+vin+'.'+date+'.part3.mat')
        dist.append(attr['distance'])
        cmean.append(attr['cmean'])
        centropy.append(attr['centropy'])
        #feature -- similarity among maps during rush hours (0: 6am-12am; 1: 12am-18pm, 2:18pm-24pm) -- on this day
        s1 = mapSim.mapsSimilarity(mat1,mat2)
        s2 = mapSim.mapsSimilarity(mat1,mat3)
        s3 = mapSim.mapsSimilarity(mat2,mat3)
        innerdatemapsim.append({'s1':s1,'s2':s2,'s3':s3})
        matsum.append(matfull)

    feat_dist_mean = np.mean(dist)
    feat_dist_entropy = stats.entropy(dist)
    feat_cmean_mean = np.mean(cmean)
    feat_centropy_mean = np.mean(centropy)
    s1arr = map(lambda x:x['s1'],innerdatemapsim)
    s2arr = map(lambda x:x['s2'],innerdatemapsim)
    s3arr = map(lambda x:x['s3'],innerdatemapsim)
    feat_s1_mean = np.mean(s1arr)
    feat_s2_mean = np.mean(s2arr)
    feat_s3_mean = np.mean(s3arr)
    feat_s1_entropy = stats.entropy(s1arr)
    feat_s2_entropy = stats.entropy(s2arr)
    feat_s3_entropy = stats.entropy(s3arr)
    for i in range(0,len(matsum)):
        for j in range(i+1,len(matsum)):
            intradatemapsim.append(mapSim.mapsSimilarity(matsum[i],matsum[j]))
    feat_intradatesim_mean = np.mean(intradatemapsim)
    feat_intradatesim_entropy = stats.entropy(intradatemapsim)
    feature = [feat_dist_mean,feat_dist_entropy,feat_cmean_mean,feat_centropy_mean,feat_s1_mean,feat_s1_entropy,feat_s2_mean,feat_s2_entropy,feat_s3_mean,feat_s3_entropy,feat_intradatesim_mean,feat_intradatesim_entropy]
    if(validate(feature)):
        feature = map(lambda x:str(x),feature)
        s = vin+","+",".join(feature)+"\n"
        outf.write(s)

#bus
busdata = {}
with open(consts['busfeaturein']) as featf:
    outf = open(consts['busfeatureout'],'w')
    for lines in featf:
        t = lines.split(",")
        vin = t[0]
        date = t[1]
        distance = t[2]
        cmean = t[3]
        centropy = t[4]
        if(vin not in busdata.keys()):
            busdata[vin]={}
        busdata[vin][date] = {'vin':vin,'date':date,'distance':float(distance),'cmean':float(cmean),'centropy':float(centropy)}

for vin in busdata.keys():
    dist = []
    cmean = []
    centropy = []
    innerdatemapsim = []
    matsum = []
    intradatemapsim = []

    for date in busdata[vin].keys():
        attr = busdata[vin][date]
        matfull = np.loadtxt(consts['busmap']+vin+'.'+date+'.full.mat')
        mat1 = np.loadtxt(consts['busmap']+vin+'.'+date+'.part1.mat')
        mat2 = np.loadtxt(consts['busmap']+vin+'.'+date+'.part2.mat')
        mat3 = np.loadtxt(consts['busmap']+vin+'.'+date+'.part3.mat')
        dist.append(attr['distance'])
        cmean.append(attr['cmean'])
        centropy.append(attr['centropy'])
        #feature -- similarity among maps during rush hours (0: 6am-12am; 1: 12am-18pm, 2:18pm-24pm) -- on this day
        s1 = mapSim.mapsSimilarity(mat1,mat2)
        s2 = mapSim.mapsSimilarity(mat1,mat3)
        s3 = mapSim.mapsSimilarity(mat2,mat3)
        innerdatemapsim.append({'s1':s1,'s2':s2,'s3':s3})
        matsum.append(matfull)
    print dist
    feat_dist_mean = np.mean(dist)
    feat_dist_entropy = stats.entropy(dist)
    feat_cmean_mean = np.mean(cmean)
    feat_centropy_mean = np.mean(centropy)
    s1arr = map(lambda x:x['s1'],innerdatemapsim)
    s2arr = map(lambda x:x['s2'],innerdatemapsim)
    s3arr = map(lambda x:x['s3'],innerdatemapsim)
    feat_s1_mean = np.mean(s1arr)
    feat_s2_mean = np.mean(s2arr)
    feat_s3_mean = np.mean(s3arr)
    feat_s1_entropy = stats.entropy(s1arr)
    feat_s2_entropy = stats.entropy(s2arr)
    feat_s3_entropy = stats.entropy(s3arr)
    for i in range(0,len(matsum)):
        for j in range(i+1,len(matsum)):
            intradatemapsim.append(mapSim.mapsSimilarity(matsum[i],matsum[j]))
    feat_intradatesim_mean = np.mean(intradatemapsim)
    feat_intradatesim_entropy = stats.entropy(intradatemapsim)
    feature = [feat_dist_mean,feat_dist_entropy,feat_cmean_mean,feat_centropy_mean,feat_s1_mean,feat_s1_entropy,feat_s2_mean,feat_s2_entropy,feat_s3_mean,feat_s3_entropy,feat_intradatesim_mean,feat_intradatesim_entropy]
    if(validate(feature)):
        feature = map(lambda x:str(x),feature)
        s = vin+","+",".join(feature)+"\n"
        outf.write(s)
