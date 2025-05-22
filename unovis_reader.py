"""
Unovis psignal keys:

0 = ref ecg
4 = cecg1
5 = cecg2
6 = cecg3
7 = cecg4
"""

import matplotlib.pyplot as plt
import wfdb
import wfdb.processing
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import MinMaxScaler
import torch
import pickle
from tqdm import tqdm

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
class unovisReader():
    def __init__(self):
        self.reference_data = {} 
        self.data_to_use = {} 
        self.labels_to_use = {} 
        self.patient_indexes = [] 
    
    def loadRecords(self):
        unovis_data = {}
        refecg = np.zeros(1)
        cecg1 = np.zeros(1)
        cecg2 = np.zeros(1)
        cecg3 = np.zeros(1)
        cecg4 = np.zeros(1)

        # read the data from the records
        exclusion_set =[70, 83, 121, 159, 179, 181, 194] 
        for record_no in tqdm(range (51, 200)):
            if record_no in exclusion_set:
                continue
            self.patient_indexes.append(record_no)

            str_record_no = str(record_no)
            unovis_data[record_no] = wfdb.rdrecord(f"data/unovis/studydata/UnoViS_BigD_{str_record_no}/UnoViS_BigD_{str_record_no}")
            refecg = unovis_data[record_no].p_signal[:,0]
            cecg1 = unovis_data[record_no].p_signal[:,4]
            cecg2 = unovis_data[record_no].p_signal[:,5]
            cecg3 = unovis_data[record_no].p_signal[:,6]
            cecg4 = unovis_data[record_no].p_signal[:,7]

            print(f"Reading record number:{record_no}")

            #band pass filtering with butterworth 
            fs = float(unovis_data[record_no].fs)
            lowcut = 0.5
            highcut = 40
            refecg = butter_bandpass_filter(refecg, lowcut, highcut, fs, order=2)
            cecg1 = butter_bandpass_filter(cecg1, lowcut, highcut, fs, order=2)
            cecg2 = butter_bandpass_filter(cecg2, lowcut, highcut, fs, order=2)
            cecg3 = butter_bandpass_filter(cecg3, lowcut, highcut, fs, order=2)
            cecg4 = butter_bandpass_filter(cecg4, lowcut, highcut, fs, order=2)

            # create segments, find peaks and classify
            
            i = 0
            WINDOW_SIZE = 120

            noisy_segments_per_record = [] 
            labels_per_record = []
            ref_segments_per_record =[]
            
            # scale each segment, then look for peaks

            while i+WINDOW_SIZE < cecg1.shape[-1]:

                ref_segment =refecg[i:i + WINDOW_SIZE].reshape(-1,1)
                cecg1_segment = cecg1[i:i + WINDOW_SIZE].reshape(-1,1)
                cecg2_segment = cecg2[i:i + WINDOW_SIZE].reshape(-1,1)
                cecg3_segment = cecg3[i:i + WINDOW_SIZE].reshape(-1,1)
                cecg4_segment = cecg4[i:i + WINDOW_SIZE].reshape(-1,1)
                
                i = i + WINDOW_SIZE

                scaler = MinMaxScaler()
                ref_segment = scaler.fit_transform(ref_segment).squeeze()
                cecg1_segment = scaler.fit_transform(cecg1_segment).squeeze()
                cecg2_segment = scaler.fit_transform(cecg2_segment).squeeze()
                cecg3_segment = scaler.fit_transform(cecg3_segment).squeeze()
                cecg4_segment = scaler.fit_transform(cecg4_segment).squeeze()

                ref_peaks = wfdb.processing.xqrs_detect(ref_segment, fs=fs, verbose=False)

               # If there is no peaks in reference ecg, skip that segment
                
                if not ref_peaks.size > 0:
                    continue 
                
               # Set the label flags for each channel
               # If there is a correctly identified peak, add that segment and skip to the next segment 
                
                cecg1_peaks = wfdb.processing.xqrs_detect(cecg1_segment, fs=fs, verbose=False)

                if not cecg1_peaks.size > 0:
                    cecg1_labels = 0
                elif (ref_peaks[0] < cecg1_peaks[0] + 5 or ref_peaks[0] > cecg1_peaks[0] - 5) and cecg1_peaks.size == ref_peaks.size :
                    cecg1_labels = 1
                    noisy_segments_per_record.append(cecg1_segment)
                    labels_per_record.append(1)
                    ref_segments_per_record.append(ref_segment)
                    continue

                else:
                    cecg1_labels = 0

                cecg2_peaks = wfdb.processing.xqrs_detect(cecg2_segment, fs=fs, verbose=False)
                
                if not cecg2_peaks.size > 0:
                    cecg2_labels = 0
                elif (ref_peaks[0] < cecg2_peaks[0] + 5 or ref_peaks[0] > cecg2_peaks[0] - 5) and cecg2_peaks.size == ref_peaks.size:
                    cecg2_labels = 1
                    noisy_segments_per_record.append(cecg2_segment)
                    labels_per_record.append(1)
                    ref_segments_per_record.append(ref_segment)
                    continue

                else:
                    cecg2_labels = 0

                cecg3_peaks = wfdb.processing.xqrs_detect(cecg3_segment, fs=fs, verbose=False)
            
                if not cecg3_peaks.size > 0:
                    cecg3_labels = 0
                elif (ref_peaks[0] < cecg3_peaks[0] + 5 or ref_peaks[0] > cecg3_peaks[0] - 5) and cecg3_peaks.size == ref_peaks.size :
                    noisy_segments_per_record.append(cecg3_segment)
                    labels_per_record.append(1)
                    cecg3_labels = 1
                    ref_segments_per_record.append(ref_segment)
                    continue

                else:
                    cecg3_labels = 0

                cecg4_peaks = wfdb.processing.xqrs_detect(cecg4_segment, fs=fs, verbose=False)
                
                if not cecg4_peaks.size > 0:
                    cecg4_labels = 0               
                elif (ref_peaks[0] < cecg4_peaks[0] + 5 or ref_peaks[0] > cecg4_peaks[0] - 5) and cecg4_peaks.size == ref_peaks.size :
                    noisy_segments_per_record.append(cecg4_segment)
                    labels_per_record.append(1)
                    cecg4_labels = 1
                    ref_segments_per_record.append(ref_segment)
                    continue

                else:
                    cecg4_labels = 0

               # If code reaches here, it means no peaks are suitable in any of the channels, so classify as UNUSABLE SEGMENT.
               # For the noisy segment, choose first channel.
            
                noisy_segments_per_record.append(cecg1_segment)
                ref_segments_per_record.append(ref_segment)
                labels_per_record.append(0)

           # for each record, push the data into the dictionaries with the key information
            self.data_to_use[record_no] = np.array(noisy_segments_per_record)
            self.reference_data[record_no] = np.array(ref_segments_per_record)
            self.labels_to_use[record_no] = np.array(labels_per_record)

            print(f"Finished Record:{record_no}. Class dist of record: {sum(labels_per_record)/len(labels_per_record)}")
            print(f"Number of added segments: {len(labels_per_record)} ")

        print(self.data_to_use.keys(), self.reference_data.keys(), self.labels_to_use.keys()) 
            

    def saveData(self):
        with open("dictionaries/final_dicts_1703/unovis_reference_ecg_by_patients.pkl", "wb") as f:
            pickle.dump(self.reference_data, f)
        
        with open("dictionaries/final_dicts_1703/unovis_cecg_by_patients.pkl", "wb") as f:
            pickle.dump(self.data_to_use, f)

        with open("dictionaries/final_dicts_1703/unovis_cecg_labels_by_patients.pkl", "wb") as f:
            pickle.dump(self.labels_to_use, f)


if __name__ == "__main__":
    reader = unovisReader()
    reader.loadRecords()
    print(reader.patient_indexes)
    reader.saveData()
