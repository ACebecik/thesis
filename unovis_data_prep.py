import matplotlib.pyplot as plt
import wfdb
import wfdb.processing
import numpy as np


unovis_data = {}
refecg = []
cecg1 = []
cecg2 = []
cecg3 = []
cecg4 = []


"""
Unovis psignal keys:

0 = ref ecg
4 = cecg1
5 = cecg2
6 = cecg3
7 = cecg4
"""

for i in range (51, 52):
    if i == 194:
        continue
    str_i = str(i)
    unovis_data[i] = wfdb.rdrecord(f"/Users/alperencebecik/Desktop/Thesis Masterfile/data/UnoViS_BigD_{str_i}/UnoViS_BigD_{str_i}")
    refecg.append(unovis_data[i].p_signal[:,0])
    cecg1.append(unovis_data[i].p_signal[:,4])
    cecg2.append(unovis_data[i].p_signal[:,5])
    cecg3.append(unovis_data[i].p_signal[:,6])
    cecg4.append(unovis_data[i].p_signal[:,7])

refecg = np.array(refecg).transpose()
cecg1 = np.array(cecg1).transpose()
cecg2 = np.array(cecg2).transpose()
cecg3 = np.array(cecg3).transpose()
cecg4 = np.array(cecg4).transpose()

print(refecg.shape, cecg1.shape, cecg2.shape, cecg3.shape, cecg4.shape)

plt.plot(cecg1, label = "cecg1")
plt.plot(refecg, label = "refecg")
plt.legend()
plt.show()


