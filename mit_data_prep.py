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



if __name__ == "__main__":

    records_file = open("/media/medit-student/Volume/alperen/repo-clone/thesis/data/physionet.org/files/mitdb/1.0.0/RECORDS")
    noise_em = wfdb.rdrecord("/media/medit-student/Volume/alperen/repo-clone/thesis/data/physionet.org/files/nstdb/1.0.0/em").p_signal

    clean_signals = {}
    noisy_signals = {}
    noise_perc = 0.3

    # Noise Preparation





    name = ""
    for char in records_file.read():
        if char == '\n':
            continue
        name = name+char
        if len(name) == 3:
            clean_signals[name] = wfdb.rdrecord(f"/media/medit-student/Volume/alperen/repo-clone/thesis/data/physionet.org/files/mitdb/1.0.0/{name}").p_signal
            noisy_signals[name] = clean_signals[name]*(1-noise_perc)  + noise_em*noise_perc

            name=""
    print(clean_signals.keys())

    records_file.close()

    # window size is 1 seconds of measurement to indicate clean / noisy fragment

    fs = 360 # sampling rate
    #window_size = 3 * fs

    WINDOW_SIZE = 1*fs
    dataset = []
    top_labels = []

    training_data = []
    target_labels = []

    # find if peaks match = usable, if not = unusable
    for key in clean_signals.keys():

        record_length = len(noisy_signals[key])
        labels = np.zeros((record_length, 1))  # 0: not usable class , 1: usable ecg class
        i = 0
        while i < record_length:
            if i + WINDOW_SIZE >= record_length:
                break
            else:
                rpeaks = wfdb.processing.xqrs_detect(clean_signals[key][i:i + WINDOW_SIZE, 0], fs=360, verbose=False)
                rpeaks_noisy = wfdb.processing.xqrs_detect(noisy_signals[key][i:i + WINDOW_SIZE, 0], fs=360, verbose=False)
                if set(rpeaks) == set(rpeaks_noisy):
                    target_labels.append(1)
                else:
                    target_labels.append(0)
            batched_train_data = noisy_signals[key][i:i + WINDOW_SIZE, 0]
            training_data.append(batched_train_data)
            i = i + WINDOW_SIZE

        print(f"Peaks done and added to the dataset for record {key}")
        print(f"Class distribution of the record: {sum(target_labels)/len(target_labels)}")

        """if key == "100":
                break """

    y = torch.Tensor(target_labels)
    X = torch.Tensor(training_data)

   #save peaks for future 
    torch.save(X, "tensors/mit_all_records_X_w360.pt")
    torch.save(y, "tensors/mit_all_records_y_w360.pt")    
