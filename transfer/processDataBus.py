import numpy as np
import matplotlib.pyplot as plt
import os
consts = {
    'taxifilepath':'../data/sjtu/Taxi/raw',
    'outputdir':'../data/sjtu/Taxi/processed/',
    'busfilepath':'../data/sjtu/Bus/raw/',
    'outputdir':'../data/sjtu/Bus/processed/',
    'leftmax' : 121.0,
    'rightmax': 121.8,
    'topmax'  : 31.4,
    'botmax'  : 30.7,
    'grid':24,
    'passweight':0.3,
    'pltdir':'../data/sjtu/Bus/img/',
}

for eachfile in os.listdir(consts['busfilepath']):
    data = {}
    cnt = 0

    with open(consts['busfilepath']+eachfile) as f:
        for lines in f:
            print cnt
            cnt +=1
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

    print "loadfinish"
    mat_dim = consts['grid']
    u_hor = (consts['rightmax']-consts['leftmax'])/mat_dim
    u_ver = (consts['topmax']-consts['botmax'])/mat_dim
    cnt = 0
    l=len(data.keys())
    for vin in data.keys():
        cnt +=1
        print "%i out of %f" %(cnt,l)
        for date in data[vin].keys():
            fname = vin+"."+date+".full.mat"

            matrix = np.zeros(shape = (mat_dim,mat_dim))
            last_lng = -1
            last_lat = -1
            c = 0
            for point in data[vin][date]:
                d_lng = int((point['lng']-consts['leftmax'])/u_hor)
                d_lat = int((consts['topmax']-point['lat'])/u_ver)
                if(d_lng<consts['grid'] and d_lng>0 and d_lat<consts['grid'] and d_lat>0):
                    if(d_lng != last_lng and d_lat != last_lat):
                        matrix[d_lat][d_lng] += consts['passweight']
                        last_lat = d_lat
                        last_lng = d_lng
                        if(matrix[d_lat][d_lng]>1):
                            matrix[d_lat][d_lng]=1

            imgfname = vin+'.'+date+".full.jpg"
            matfname = vin+'.'+date+".full.mat"
            plt.imsave(consts['pltdir']+imgfname,matrix)
            np.savetxt(consts['outputdir']+matfname,matrix,fmt='%10.2f')

