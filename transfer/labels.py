import os
consts = {
    'busdir':'../data/sjtu/Bus/processed/',
    'taxidir':'../data/sjtu/Taxi/processed/',
    'outdir':'../data/sjtu/label.csv',
    'uber':'0',
    'private':'1',
}

busarr = os.listdir(consts['busdir'])
taxiarr = os.listdir(consts['taxidir'])
written = []

with open(consts['outdir'],'w') as f:
    for each in busarr:
        vin = each.split(".")[0]
        if(vin not in written):
            written.append(vin)
            s = str(vin)+","+consts['private']+"\n"
            f.write(s)
    for each in taxiarr:
        vin = each.split(".")[0]
        if(vin not in written):
            written.append(vin)
            s = str(vin)+","+consts['uber']+"\n"
            f.write(s)
f.close()