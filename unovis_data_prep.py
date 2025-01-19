import matplotlib.pyplot as plt
import wfdb
import wfdb.processing
import numpy as np


unovis_data = {}
refecg = np.zeros(1)
cecg1 = np.zeros(1)
cecg2 = np.zeros(1)
cecg3 = np.zeros(1)
cecg4 = np.zeros(1)


"""
Unovis psignal keys:

0 = ref ecg
4 = cecg1
5 = cecg2
6 = cecg3
7 = cecg4
"""

for i in range (51, 53):
    if i == 194:
        continue
    str_i = str(i)
    unovis_data[i] = wfdb.rdrecord(f"//media/medit-student/Volume/alperen/repo-clone/thesis/data/unovis/studydata/UnoViS_BigD_{str_i}/UnoViS_BigD_{str_i}")
    refecg = np.concatenate((refecg, unovis_data[i].p_signal[:,0]))
    cecg1 = np.concatenate((cecg1, unovis_data[i].p_signal[:,4]))
    cecg2 = np.concatenate((cecg2, unovis_data[i].p_signal[:,5]))
    cecg3 = np.concatenate((cecg3, unovis_data[i].p_signal[:,6]))
    cecg4 = np.concatenate((cecg4, unovis_data[i].p_signal[:,7]))

print(refecg.shape, cecg1.shape, cecg2.shape, cecg3.shape, cecg4.shape)

"""plt.plot(cecg1, label = "cecg1")
plt.plot(refecg, label = "refecg")
plt.legend()
plt.savefig("plots/temp-unovis")"""


