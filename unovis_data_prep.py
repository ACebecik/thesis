import matplotlib.pyplot as plt
import wfdb
import wfdb.processing
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import MinMaxScaler
import torch

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

data_to_use = {} 
labels_to_use ={} 

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
for record_no in range (51, 200):
    if record_no in exclusion_set:
        continue
    str_record_no = str(record_no)
    unovis_data[record_no] = wfdb.rdrecord(f"//media/medit-student/Volume/alperen/repo-clone/thesis/data/unovis/studydata/UnoViS_BigD_{str_record_no}/UnoViS_BigD_{str_record_no}")
    refecg = unovis_data[record_no].p_signal[:,0]
    cecg1 = unovis_data[record_no].p_signal[:,4]
    cecg2 = unovis_data[record_no].p_signal[:,5]
    cecg3 = unovis_data[record_no].p_signal[:,6]
    cecg4 = unovis_data[record_no].p_signal[:,7]

    print(f"Reading record number:{record_no}")

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
        
        scaler = MinMaxScaler()
        ref_segment = scaler.fit_transform(ref_segment).squeeze()
        cecg1_segment = scaler.fit_transform(cecg1_segment).squeeze()
        cecg2_segment = scaler.fit_transform(cecg2_segment).squeeze()
        cecg3_segment = scaler.fit_transform(cecg3_segment).squeeze()
        cecg4_segment = scaler.fit_transform(cecg4_segment).squeeze()

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

    print(f"Class Distribution of Channels for Record: {record_no}")
    cd1 = sum(cecg1_labels)/len(cecg1_labels)
    cd2 = sum(cecg2_labels)/len(cecg2_labels)
    cd3 = sum(cecg3_labels)/len(cecg3_labels)
    cd4 = sum(cecg4_labels)/len(cecg4_labels)

    cd =[cd1, cd2, cd3, cd4]
    print(f" Class Distributions of Channels:{cd}")
    selected_channel = np.argmax(cd)
    print (f"Selected Channel:{selected_channel + 1}")
    
    cecg1_labels = np.array(cecg1_labels)
    cecg2_labels = np.array(cecg2_labels)
    cecg3_labels = np.array(cecg3_labels)
    cecg4_labels = np.array(cecg4_labels)

    if selected_channel == 0:
        data_to_use[record_no] = cecg1
        labels_to_use[record_no] = cecg1_labels

    elif selected_channel == 1:
        data_to_use[record_no] = cecg2
        labels_to_use[record_no] = cecg2_labels

    elif selected_channel == 2:
        data_to_use[record_no] = cecg3
        labels_to_use[record_no] = cecg3_labels

    else:
        data_to_use[record_no] = cecg4
        labels_to_use[record_no] = cecg4_labels

X_total=[]
y_total=np.array([0])

for key in data_to_use.keys():
    i = 0
    while i + WINDOW_SIZE < data_to_use[key].shape[-1]:
        X_total.append(data_to_use[key][i:i + WINDOW_SIZE])

        i = i + WINDOW_SIZE
    y_total = np.concatenate((y_total, labels_to_use[key])) 

y_total = np.delete(y_total,[0])
X_tensor = torch.Tensor(X_total)
y_tensor = torch.Tensor(y_total)

torch.save(X_tensor, "tensors/unovis_all_records_X_w120.pt")
torch.save(y_tensor, "tensors/unovis_all_records_y_w120.pt")





