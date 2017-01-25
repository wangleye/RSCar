import numpy as np
import matplotlib.pyplot as plt
import os
import operator
import math
import mapSim as mapSim
import scipy.stats as stats

consts = {
    'taxifilepath':'../data/sjtu/Taxi/raw/',
    'outputdir':'../data/sjtu/oneclass/feature_ly.txt',
    'busfilepath':'../data/sjtu/Bus/raw/',
    'taximap':'../data/sjtu/oneclass/tmap/',
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

mat_dim = consts['grid']
u_hor = (consts['rightmax']-consts['leftmax'])/mat_dim
u_ver = (consts['topmax']-consts['botmax'])/mat_dim


def smoothSeries(s,lower,upper):
    prefix = int(consts['smoothwindow']/2)
    for i in range(prefix+1,len(s)):
        slice = s[i-prefix:i+prefix+1]
        mean = np.mean(slice)
        std = np.std(slice)
        if not(lower<s[i]<upper and abs(mean-s[i])<1*std):
            s[i] = mean
    return s

def validate(feature):
    #feature = [feat_dist_mean,feat_dist_entropy,feat_s1_mean,feat_s1_entropy,feat_s2_mean,feat_s2_entropy,feat_s3_mean,feat_s3_entropy,feat_intradatesim_mean,feat_intradatesim_entropy]
    dist = 10<feature[0]<1000
    nan = (np.sum(np.isnan(feature)) == 0)
    if(dist and nan):
        return True
    else:
        return False


def process(data,f,basepath):
    l = len(data.keys())
    cnt=0
    print "totally"+str(l)+"keys"
    for vin in data.keys():
        cnt+=1
        if(cnt%2000==0):
            print cnt
            return # only use 2000 cars 
        for date in data[vin].keys():
            records = sorted(data[vin][date],key = operator.itemgetter('time'))
            data[vin][date] = records
            latarr = map(lambda x:x['lat'],records)
            lngarr = map(lambda x:x['lng'],records)
            #smooth
            smoothlat = smoothSeries(latarr,consts['botmax'],consts['topmax'])
            smoothlng = smoothSeries(lngarr,consts['leftmax'],consts['rightmax'])
            distanceday = 0
            distance6 = 0
            distance12 = 0
            distance18 = 0
            distances = [distance6, distance12, distance18]
            mallday = np.zeros(shape = (mat_dim,mat_dim))
            m6 = np.zeros(shape = (mat_dim,mat_dim))
            m12 = np.zeros(shape = (mat_dim,mat_dim))
            m18 = np.zeros(shape = (mat_dim,mat_dim))
            marr = [m6,m12,m18]
            for i in range(0,len(latarr)):
                data[vin][date][i]['lat'] = smoothlat[i]
                data[vin][date][i]['lng'] = smoothlng[i]
                #feature -- distance -- on this day
                if(i>1):
                    latdist = (smoothlat[i]-smoothlat[i-1])*consts['kmperlat']
                    lngdist = (smoothlng[i]-smoothlng[i-1])*consts['kmperlng']
                    dis = math.sqrt(latdist*latdist + lngdist*lngdist)
                    t = int(data[vin][date][i]['time'].split(":")[0])/6-1
                    distances[t] += dis
                #feature -- daily coverage -- on this day
                d_lng = int((smoothlng[i]-consts['leftmax'])/u_hor)
                d_lat = int((consts['topmax']-smoothlat[i])/u_ver)
                if(d_lng<consts['grid'] and d_lng>0 and d_lat<consts['grid'] and d_lat>0):
                    t = int(data[vin][date][i]['time'].split(":")[0])/6-1
                    if(t>0):
                        try:
                            marr[t][d_lat][d_lng] = 1
                        except Exception,e:
                            print t
                            print d_lat
                            print d_lng
            mallday = marr[0]+marr[1]+marr[2]
            coverage = np.sum(mallday)
            c1 = np.sum(marr[0])
            c2 = np.sum(marr[1])
            c3 = np.sum(marr[2])
            c = [c1,c2,c3]
            cmean = np.mean(c)
            c_diff = mapSim.mapsSimilarity(marr[0],marr[2])
            try:
                centropy = stats.entropy(c)
            except Exception,e:
                print c
            writestr = map(lambda x:str(x),[vin,date,distances[0],distances[2],c[0],c[2],c_diff])
            tbwrite = ",".join(writestr)+"\n"
            f.write(tbwrite)
            # m1path = basepath+str(vin)+"."+str(date)+".part1.mat"
            # m2path = basepath+str(vin)+"."+str(date)+".part2.mat"
            # m3path = basepath+str(vin)+"."+str(date)+".part3.mat"
            # m4path = basepath+str(vin)+"."+str(date)+".full.mat"
            # np.savetxt(m1path,marr[0],fmt='%10.2f')
            # np.savetxt(m2path,marr[1],fmt='%10.2f')
            # np.savetxt(m3path,marr[2],fmt='%10.2f')
            # np.savetxt(m4path,mallday,fmt='%10.2f')
def main():
    outfile = open(consts['outputdir'],'w')

    print 'loading file'
    cnt = 0
    print 'loading taxi'
    '''


    for eachfile in os.listdir(consts['busfilepath']):
        cnt +=1
        print cnt
        with open(consts['busfilepath']+eachfile) as f:
            data = {}
            for lines in f:
                tuples = lines.split(",")
                vin = tuples[2]
                lng = tuples[6]
                lat = tuples[7]
                time = tuples[12].split(" ")
                date = time[0]
                time = time[1]
                valid = tuples[14]
                if(int(valid) == 0):
                    if(vin not in data.keys()):
                        data[vin] = {}
                    if(date not in data[vin].keys()):
                        data[vin][date] = []
                    data[vin][date].append({'vin':vin,'lng':float(lng),'lat':float(lat),'time':str(time),'date':date})
            process(data,outfile,consts['busmap'])
    '''

    print 'loading taxi'
    for eachfile in os.listdir(consts['taxifilepath']):
        cnt +=1
        print 'file', cnt
        with open(consts['taxifilepath']+eachfile) as f:
            data = {}
            cnt2 = 0
            for lines in f:
                cnt2 += 1
                if cnt2%10000==0:
                    print 'line', cnt2
                tuples = lines.split(",")
                vin = tuples[1]
                lng = tuples[2]
                lat = tuples[3]
                time = tuples[6].split(" ")
                date = time[0]
                time = time[1]
                if(vin not in data.keys()):
                    data[vin] = {}
                if(date not in data[vin].keys()):
                    data[vin][date] = []
                data[vin][date].append({'vin':vin,'lng':float(lng),'lat':float(lat),'time':str(time),'date':date})
            process(data,outfile,consts['taximap'])
    f.close()

main()