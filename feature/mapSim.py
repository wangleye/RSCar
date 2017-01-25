from itertools import combinations
import numpy as np
from scipy.ndimage import convolve
from scipy import stats


consts = {
    "fullmapweight":2,
    "partmapweight":1,
    "rawmapweight":0.4,
    "convmapweight":0.3,
    "partmapweight":0.3,
}
def sim(maps,topK = 3):
    s = []
    maps = filter(lambda x:np.sum(x['full'])>0,maps)
    if(len(maps)<2):
        return 0
    for i in combinations(maps,2):
        m1p = i[0]['part']
        m1f = i[0]['full']
        m2p = i[1]['part']
        m2f = i[1]['full']
        sf = mapsSimilarity(m1f,m2f)
        sp = []
        for m in zip(m1p,m2p):
            sp.append(mapsSimilarity(m[0],m[1]))
        s.append(sf*consts['fullmapweight']+np.nansum(sp)*consts['partmapweight'])
    sim_score = np.nanmean(s)
    sim_entropy = stats.entropy(s)
    s.sort()
    highest = s[topK*(-1):]
    h_score = np.mean(highest)
    return {'mean':sim_score,'entropy':sim_entropy,'highest':h_score}

def mapsSimilarity(m1,m2):
    s1 = mapJaccard(m1,m2)
    s2 = mapJaccard(mapShift(m1,0),m2)
    s3 = mapJaccard(mapShift(m1,1),m2)
    s4 = mapJaccard(mapShift(m1,2),m2)
    s5 = mapJaccard(mapShift(m1,3),m2)
    sim1 = np.nanmean([s1,s2,s3,s4,s5])
    cm2 = conv(m2)
    s1 = mapJaccard(conv(m1),cm2)
    s2 = mapJaccard(conv(mapShift(m1,0)),cm2)
    s3 = mapJaccard(conv(mapShift(m1,1)),cm2)
    s4 = mapJaccard(conv(mapShift(m1,2)),cm2)
    s5 = mapJaccard(conv(mapShift(m1,3)),cm2)
    sim2 = np.nanmean([s1,s2,s3,s4,s5])
    pm2 = pool(m2)
    s1 = mapJaccard(pool(m1),pm2)
    s2 = mapJaccard(pool(mapShift(m1,0)),pm2)
    s3 = mapJaccard(pool(mapShift(m1,1)),pm2)
    s4 = mapJaccard(pool(mapShift(m1,2)),pm2)
    s5 = mapJaccard(pool(mapShift(m1,3)),pm2)
    sim3 = np.nanmean([s1,s2,s3,s4,s5])
    return consts['rawmapweight']*sim1+consts['convmapweight']*sim2+consts['partmapweight']*sim3

def pool(m,neighborhood=2):
    if(m.shape[0]!=m.shape[1] or m.shape[0]%neighborhood!=0):
        return 0
    dim = m.shape[0]/neighborhood
    ret = np.zeros((dim,dim))
    for i in range(0,dim):
        for j in range(0,dim):
            ret[i][j] = np.nanmax(m[2*i:2*(i+1),2*j:2*(j+1)])
    return ret

def conv(m):
    laplacian = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    return convolve(m,laplacian,mode="constant",cval = 0.0)

def mapJaccard(m1,m2):
    m1 = m1>0
    m2 = m2>0
    max = np.maximum(m1,m2)
    min = np.minimum(m1,m2)
    sim = float(min.sum()+0.1)/(max.sum()+0.1)
    return sim

def mapShift(map,op):
    if(op==0):#up
        r = np.roll(map,-1,axis = 0)
        r[-1] = 0
    if(op==1):#down
        r = np.roll(map,1,axis = 0)
        r[0] = 0
    if(op==2):#left
        r = np.roll(map,-1,axis = 1)
        r[:,-1] = 0
    if(op==3):#right
        r = np.roll(map,1,axis = 1)
        r[:,0] = 0
    return r