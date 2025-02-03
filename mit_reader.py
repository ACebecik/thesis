import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wfdb
import wfdb.processing
import scipy
import time
import math
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.metrics import classification_report
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import butter, lfilter



def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

class mitReader():
    def __init__(self):

        self.clean_signals = {}
        self.noisy_signals = {}
        self.data_to_use = {}
        self.labels_to_use = {}  
        self.reference_data = {}
        self.patient_indexes = [] 

    def loadRecords(self):
        records_file = open("/media/medit-student/Volume/alperen/repo-clone/thesis/data/physionet.org/files/mitdb/1.0.0/RECORDS")
        noise_em = wfdb.rdrecord("/media/medit-student/Volume/alperen/repo-clone/thesis/data/physionet.org/files/nstdb/1.0.0/em").p_signal
        noise_perc = 0.3

        name = ""
        for char in records_file.read():
            if char == '\n':
                continue
            name = name+char
            if len(name) == 3:
                self.clean_signals[name] = wfdb.rdrecord(f"/media/medit-student/Volume/alperen/repo-clone/thesis/data/physionet.org/files/mitdb/1.0.0/{name}").p_signal
                self.noisy_signals[name] = self.clean_signals[name]*(1-noise_perc)  + noise_em*noise_perc

                name=""

        print(self.clean_signals.keys())
        records_file.close()


        fs = 360 # sampling rate
        WINDOW_SIZE = 1*fs
        dataset = []
        top_labels = []

        training_data = []
        target_labels = []
        clean_data = [] 



        # find if peaks match = usable, if not = unusable
        for key in tqdm (self.clean_signals.keys()):

            print(f"Reading record number:{key}")
            self.patient_indexes.append(int(key))
            print(f"patient indexes:{self.patient_indexes}  ")
            #band pass filtering with butterworth 
            fs = 360.0
            lowcut = 0.5
            highcut = 40
            self.clean_signals[key] = butter_bandpass_filter(self.clean_signals[key], lowcut, highcut, fs, order=2)
            self.noisy_signals[key] = butter_bandpass_filter(self.noisy_signals[key], lowcut, highcut, fs, order=2)
            
            record_length = len(self.clean_signals[key])
            labels = np.zeros((record_length, 1))  # 0: not usable class , 1: usable ecg class
            i = 0
            while i < record_length:
                if i + WINDOW_SIZE >= record_length:
                    break
                else:
                    clean_segment = self.clean_signals[key][i:i + WINDOW_SIZE, 0].reshape(-1,1)
                    noisy_segment = self.noisy_signals[key][i:i + WINDOW_SIZE, 0].reshape(-1,1)

                    #scaling
                    scaler = MinMaxScaler()
                    clean_segment = scaler.fit_transform(clean_segment).squeeze()
                    noisy_segment = scaler.fit_transform(noisy_segment).squeeze()   

                    rpeaks = wfdb.processing.xqrs_detect(clean_segment, fs=360, verbose=False)
                    rpeaks_noisy = wfdb.processing.xqrs_detect(noisy_segment, fs=360, verbose=False)
                    if set(rpeaks) == set(rpeaks_noisy):
                        target_labels.append(1)
                    else:
                        target_labels.append(0)

               # downsample the segments
                noisy_segment_np = np.array(noisy_segment)
                noisy_segment_np = noisy_segment_np[::3]
    
                clean_segment_np = np.array(clean_segment)
                clean_segment_np = clean_segment_np[::3]

                training_data.append(noisy_segment_np)
                clean_data.append(clean_segment_np)

                i = i + WINDOW_SIZE

            print(f"Peaks done and added to the dataset for record {key}")
            print(f"Class distribution of the record: {sum(target_labels)/len(target_labels)}")

            training_data_np = np.array(training_data)
            target_labels_np = np.array(target_labels)
            clean_data_np = np.array(clean_data)
            
            self.data_to_use[int(key)] = training_data_np
            self.labels_to_use[int(key)] = target_labels_np
            self.reference_data[int(key)] = clean_data_np 

            print(self.data_to_use.keys(), self.reference_data.keys(), self.labels_to_use.keys())

    def saveData(self):
        with open("dictionaries/mit_reference_ecg_by_patients.pkl", "wb") as f:
            pickle.dump(self.reference_data, f)
        
        with open("dictionaries/mit_noisy_ecg_by_patients.pkl", "wb") as f:
            pickle.dump(self.data_to_use, f)

        with open("dictionaries/mit_noisy_ecg_labels_by_patients.pkl", "wb") as f:
            pickle.dump(self.labels_to_use, f)



if __name__ == "__main__":
    reader = mitReader()
    reader.loadRecords()
    print(reader.patient_indexes)
    reader.saveData()
