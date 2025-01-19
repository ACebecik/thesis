import matplotlib.pyplot as plt
import wfdb
import wfdb.processing
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import MinMaxScaler

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


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

# read the data from the records
exclusion_set =[62, 66, 70, 79, 83, 86, 94, 99, 103, 121, 122, 129, 146, 147, 159, 174, 179, 181, 187, 189, 194] 
for i in range (51, 70):
    if i in exclusion_set:
        continue
    str_i = str(i)
    unovis_data[i] = wfdb.rdrecord(f"//media/medit-student/Volume/alperen/repo-clone/thesis/data/unovis/studydata/UnoViS_BigD_{str_i}/UnoViS_BigD_{str_i}")
    refecg = np.concatenate((refecg, unovis_data[i].p_signal[:,0]))
    cecg1 = np.concatenate((cecg1, unovis_data[i].p_signal[:,4]))
    cecg2 = np.concatenate((cecg2, unovis_data[i].p_signal[:,5]))
    cecg3 = np.concatenate((cecg3, unovis_data[i].p_signal[:,6]))
    cecg4 = np.concatenate((cecg4, unovis_data[i].p_signal[:,7]))

print(refecg.shape, cecg1.shape, cecg2.shape, cecg3.shape, cecg4.shape)

#band pass filtering with butterworth 
fs = 360.0
lowcut = 0.5
highcut = 40
cecg1 = butter_bandpass_filter(cecg1, lowcut, highcut, fs, order=2)
cecg2 = butter_bandpass_filter(cecg2, lowcut, highcut, fs, order=2)
cecg3 = butter_bandpass_filter(cecg3, lowcut, highcut, fs, order=2)
cecg4 = butter_bandpass_filter(cecg4, lowcut, highcut, fs, order=2)

# create segments, find peaks and classify
fs = 360
i = 0
WINDOW_SIZE = 120

cecg1_labels =[]
cecg2_labels =[]
cecg3_labels =[]
cecg4_labels =[]

# scale each segment, then look for peaks

while i+WINDOW_SIZE < cecg1.shape[-1]:

    ref_segment =refecg[i:i + WINDOW_SIZE].reshape(-1,1)
    cecg1_segment = cecg1[i:i + WINDOW_SIZE].reshape(-1,1)
    cecg2_segment = cecg2[i:i + WINDOW_SIZE].reshape(-1,1)
    cecg3_segment = cecg3[i:i + WINDOW_SIZE].reshape(-1,1)
    cecg4_segment = cecg4[i:i + WINDOW_SIZE].reshape(-1,1)
    
    """    plt.plot(ref_segment, label = "ref before")
    plt.plot(cecg1_segment, label = "cecg1 before")"""

    scaler = MinMaxScaler()
    ref_segment = scaler.fit_transform(ref_segment).squeeze()
    cecg1_segment = scaler.fit_transform(cecg1_segment).squeeze()
    cecg2_segment = scaler.fit_transform(cecg2_segment).squeeze()
    cecg3_segment = scaler.fit_transform(cecg3_segment).squeeze()
    cecg4_segment = scaler.fit_transform(cecg4_segment).squeeze()

    """    plt.plot(ref_segment, label = "ref after")
    plt.plot(cecg1_segment, label = "cecg1 after")
    plt.legend()
    plt.title("Before after scaling in segment")
    plt.savefig("plots/scaler try")
    plt.clf()"""

    ref_peaks = wfdb.processing.xqrs_detect(ref_segment, fs=360, verbose=False)
    cecg1_peaks = wfdb.processing.xqrs_detect(cecg1_segment, fs=360, verbose=False)

    if set(ref_peaks) == set(cecg1_peaks):
        cecg1_labels.append(1)
    else:
        cecg1_labels.append(0)

    cecg2_peaks = wfdb.processing.xqrs_detect(cecg2_segment, fs=360, verbose=False)
    
    if set(ref_peaks) == set(cecg2_peaks):
        cecg2_labels.append(1)
    else:
        cecg2_labels.append(0)

    cecg3_peaks = wfdb.processing.xqrs_detect(cecg3_segment, fs=360, verbose=False)
   
    if set(ref_peaks) == set(cecg3_peaks):
        cecg3_labels.append(1)
    else:
        cecg3_labels.append(0)

    cecg4_peaks = wfdb.processing.xqrs_detect(cecg4_segment, fs=360, verbose=False)
    
    if set(ref_peaks) == set(cecg4_peaks):
        cecg4_labels.append(1)
    else:
        cecg4_labels.append(0)

    i = i + WINDOW_SIZE

print(f"Class distribution of the cecg1: {sum(cecg1_labels)/len(cecg1_labels)}")
print(f"Class distribution of the cecg2: {sum(cecg2_labels)/len(cecg2_labels)}")
print(f"Class distribution of the cecg3: {sum(cecg3_labels)/len(cecg3_labels)}")
print(f"Class distribution of the cecg4: {sum(cecg4_labels)/len(cecg4_labels)}")

#print(f"Number of peaks found: {len(ref_peaks)}")


