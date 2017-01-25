import os
import numpy as np
import matplotlib.pyplot as plt

base = '../data/sjtu/Taxi/processed/'
mat = os.listdir(base)
m = np.zeros(shape = (24,24))
c = 0
print len(mat)
for each in mat:
    print c
    c+=1
    m += np.loadtxt(base+each).reshape(24,24)

plt.gray()
np.savetxt('./tmat',m,fmt='%10.2f')
print m