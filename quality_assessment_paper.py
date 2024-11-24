"""
This file implements the quality assessment paper:
Noise Detection in Electrocardiography Signal for Robust Heart Rate Variability Analysis: A Deep Learning Approach
by Ansari et al.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import wfdb
import wfdb.processing
import scipy
import time

class NoiseDetector(nn.Module):
    def __init__(self, in_channels):
        super(NoiseDetector,self).__init__()

        # First set of Conv,Relu,Pooling,Dropout
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=7, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=0.7)

        # 2nd
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=0.7)

        #3rd
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(p=0.7)

        #4th
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(p=0.7)

        #FC layer and softmax
        self.fc1 = nn.Linear(in_features=1600, out_features=1024)
        self.relu5 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=1024, out_features=2)
        self.logSoftmax = nn.LogSoftmax()


    def forward(self, x):
        "Define the forward pass"

        #layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        #layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        #layer 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        #layer 4
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.drop4(x)

        x = self.fc1(x)
        x = self.relu5(x)

        x = self.fc2(x)
        output = self.logSoftmax(x)

        return output


if __name__ == "__main__":

    records_file = open("/Users/alperencebecik/Desktop/Thesis Masterfile/data/mit-bih-arrhythmia-database-1.0.0/RECORDS")
    records = {}
    name = ""
    for char in records_file.read():
        if char == '\n':
            continue
        name = name+char
        if len(name) == 3:
            records[name] = wfdb.rdrecord(f"/Users/alperencebecik/Desktop/Thesis Masterfile/data/mit-bih-arrhythmia-database-1.0.0/{name}")
            name=""
    print(records.keys())

    records_file.close()

    noise_record = wfdb.rdrecord("/Users/alperencebecik/Desktop/Thesis Masterfile/data/mit-bih-noise-stress-test-database-1.0.0/em")
    print(noise_record.p_signal.shape, records['100'].p_signal.shape)

    noisy_signal = records['100'].p_signal*4/5 + noise_record.p_signal*1/5

    """    plt.plot(records['100'].p_signal[:,0], label = 'clean', )
        plt.plot(noisy_signal[:,0], label = 'noisy')
        plt.legend()
        plt.show()
    
        rpeaks = wfdb.processing.xqrs_detect(records['100'].p_signal[:, 0], fs=360, verbose=False)
    
        rpeaks_noisy = wfdb.processing.xqrs_detect(noisy_signal[:,0], fs=360, verbose=False)
        plt.plot(noisy_signal[:,0])
        plt.plot(rpeaks, records['100'].p_signal[:, 0][rpeaks], "x")
        plt.show()
    
        print(rpeaks_noisy.shape, rpeaks.shape)"""

    # window size is 3 seconds of measurement to indicate clean / noisy fragment

    fs = 360 # sampling rate
    window_size = 3 * fs
    record_length = len(noisy_signal[:,0])
    labels = np.zeros(record_length) # 0: notusable, 1:usable

    # find if peaks match = usable, if not = unusable
    i = 0
    while i < record_length:
        if i + window_size >= record_length:
            rpeaks = wfdb.processing.xqrs_detect(records['100'].p_signal[i:, 0], fs=360, verbose=False)
            rpeaks_noisy = wfdb.processing.xqrs_detect(noisy_signal[i:, 0], fs=360, verbose=False)
            if set(rpeaks) == set(rpeaks_noisy):
                labels[i:] = 1
            break
        else:
            rpeaks = wfdb.processing.xqrs_detect(records['100'].p_signal[i:i+window_size, 0], fs=360, verbose=False)
            rpeaks_noisy = wfdb.processing.xqrs_detect(noisy_signal[i:i+window_size, 0], fs=360, verbose=False)
            if set(rpeaks) == set(rpeaks_noisy):
                labels[i:i+window_size] = 1

        i = i + window_size



    # define training hyperparameters
    INIT_LR = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 10
    # define the train and val splits
    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 1 - TRAIN_SPLIT
    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("Initializing model...")
    model = NoiseDetector(in_channels=1).to(device)
    opt = optim.Adam(model.parameters(), lr=INIT_LR)
    lossFn = nn.NLLLoss()

    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()





