import numpy as np
import matplotlib.pyplot as plt
import os
import operator
import math
import mapSim as mapSim
import scipy.stats as stats

consts = {
    'workday' : ["2016-4-29","2016-5-3","2016-5-4","2016-5-5","2016-5-6","2016-5-9","2016-5-10","2016-5-11"],
    'featureout':'../data/sjtu/oneclass/200labeledfeature.txt',
    'rawfeature':'../feature/feature_extracted.txt',
    'fullmap':'../feature/map/full/',
    'partmap':'../feature/map/part/',
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

#        feature = [num_of_days,mean_num_of_records,mean_distance,entropy_distance,mean_fuel,entropy_fuel,long_stop,short_stop,hor_ex_all,hor_ex_daily,ver_ex_all,ver_ex_daily,meanScore,highScore,entropyScore]
with open(consts['rawfeature']) as rawf:
    outf = open(consts['featureout'],'w')
    for lines in rawf:
        t = lines.split(":")
        vin = t[0]
        existingfeatures = t[1].split(",")
        feat_dist_mean = existingfeatures[2]
        feat_dist_entropy = existingfeatures[3]
        innerdatemapsim = []
        coverage_mean = []
        coverage_entropy = []
        matsum = []
        intradatemapsim = []
        for date in consts['workday']:
            try:
                matfull = np.loadtxt(consts['fullmap']+vin+'.'+date+'.full.mat')
            except Exception,e:
                matfull = np.zeros(shape = (consts['grid'],consts['grid']))
            try:
                mat1 = np.loadtxt(consts['partmap']+vin+'.'+date+'.part1.mat')
            except Exception,e:
                mat1 = np.zeros(shape = (consts['grid'],consts['grid']))
            try:
                mat2 = np.loadtxt(consts['partmap']+vin+'.'+date+'.part2.mat')
            except Exception,e:
                mat2 = np.zeros(shape = (consts['grid'],consts['grid']))
            try:
                mat3 = np.loadtxt(consts['partmap']+vin+'.'+date+'.part3.mat')
            except Exception,e:
                mat3 = np.zeros(shape = (consts['grid'],consts['grid']))
            s1 = mapSim.mapsSimilarity(mat1,mat2)
            s2 = mapSim.mapsSimilarity(mat1,mat3)
            s3 = mapSim.mapsSimilarity(mat2,mat3)
            innerdatemapsim.append({'s1':s1,'s2':s2,'s3':s3})
            matsum.append(matfull)
            c1 = np.sum(mat1>0)
            c2 = np.sum(mat2>0)
            c3 = np.sum(mat3>0)
            c = [c1,c2,c3]
            cmean = np.mean(c)
            coverage_mean.append(cmean)
            centropy = stats.entropy(c)
            coverage_entropy.append(centropy)
        feat_cmean_mean = np.mean(coverage_mean)
        feat_centropy_mean = np.mean(coverage_entropy)
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
        feature = map(lambda x:'0' if(np.isnan(float(x))) else str(x),feature)
        print feature

        s = vin+","+",".join(feature)+"\n"
        outf.write(s)