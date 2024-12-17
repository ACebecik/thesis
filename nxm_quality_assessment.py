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
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.metrics import classification_report
import argparse
import pickle



class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index])
        y = torch.Tensor(self.y[index])
        return (x, y)

    def __len__(self):
        count = len(self.x)
        return count


class NoiseDetector(nn.Module):
    def __init__(self, in_channels):
        super(NoiseDetector,self).__init__()

        # First set of Conv,Relu,Pooling,Dropout
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=0.1)

        # 2nd
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=0.1)

        #3rd
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(p=0.1)

        #4th
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(p=0.1)

        #FC layer and softmax
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1536, out_features=1024)
        self.relu5 = nn.ReLU()

        #self.flatten2 = nn.Flatten()
        self.fc2 = nn.Linear(in_features=1024, out_features=1)


    def forward(self, x):
        "Define the forward pass"

        #print('before conv1 layer x.shape:', x.shape)

        #layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        #print('before conv2 layer x.shape:', x.shape)

        #layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        #print('before conv3 layer x.shape:', x.shape)

        #layer 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        #print('before conv4 layer x.shape:', x.shape)

        #layer 4
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.drop4(x)

        #print('before linear1 layer x.shape:', x.shape)
        x = self.flatten1(x)

        #print('after flatten linear1 layer x.shape:', x.shape)

        x = self.fc1(x)
        x = self.relu5(x)

        #print('before linear2 layer x.shape:', x.shape)

        #x = self.flatten2(x)
        x = self.fc2(x)
        #print('after linear2 layer x.shape:', x.shape)
        #output = self.logSoftmax(x)"""
        """        x = torch.unsqueeze(x, dim = 0)
                    x = x.repeat(1,384,1)"""

        return x


if __name__ == "__main__":

    records_file = open("/Users/alperencebecik/Desktop/Thesis Masterfile/data/mit-bih-arrhythmia-database-1.0.0/RECORDS")
    noise_em = wfdb.rdrecord("/Users/alperencebecik/Desktop/Thesis Masterfile/data/mit-bih-noise-stress-test-database-1.0.0/em").p_signal

    clean_signals = {}
    noisy_signals = {}
    noise_perc = 0.3


    name = ""
    for char in records_file.read():
        if char == '\n':
            continue
        name = name+char
        if len(name) == 3:
            clean_signals[name] = wfdb.rdrecord(f"/Users/alperencebecik/Desktop/Thesis Masterfile/data/mit-bih-arrhythmia-database-1.0.0/{name}").p_signal
            noisy_signals[name] = clean_signals[name]*(1-noise_perc)  + noise_em*noise_perc
            name=""
    print(clean_signals.keys())

    records_file.close()

    # window size is 3 seconds of measurement to indicate clean / noisy fragment

    fs = 360 # sampling rate
    #window_size = 3 * fs


    window_size = 384
    dataset = []
    top_labels = []

    training_data = []
    target_labels = []
    """
        # find if peaks match = usable, if not = unusable
        for key in clean_signals.keys():
    
            record_length = len(noisy_signals[key])
            labels = np.zeros((record_length, 1))  # 0: not usable class , 1: usable ecg class
            i = 0
            while i < record_length:
                if i + window_size >= record_length:
                    break
                else:
                    rpeaks = wfdb.processing.xqrs_detect(clean_signals[key][i:i+window_size, 0], fs=360, verbose=False)
                    rpeaks_noisy = wfdb.processing.xqrs_detect(noisy_signals[key][i:i+window_size, 0], fs=360, verbose=False)
                    if set(rpeaks) == set(rpeaks_noisy):
                        target_labels.append(1)
                    else:
                        target_labels.append(0)
                batched_train_data = noisy_signals[key][i:i+window_size, 0]
                training_data.append(batched_train_data)
                i = i + window_size
    
            print(f"Peaks done and added to the dataset for record {key}")
    
    """

    #find if peaks match = usable, if not = unusable
    key = '100'

    record_length = len(noisy_signals[key])
    labels = np.zeros((record_length, 1))  # 0: not usable class , 1: usable ecg class
    i = 0
    while i < record_length:
        if i + window_size >= record_length:
            break
        else:
            rpeaks = wfdb.processing.xqrs_detect(clean_signals[key][i:i + window_size, 0], fs=360, verbose=False)
            rpeaks_noisy = wfdb.processing.xqrs_detect(noisy_signals[key][i:i + window_size, 0], fs=360, verbose=False)
            if set(rpeaks) == set(rpeaks_noisy):
                target_labels.append(1)
            else:
                target_labels.append(0)
        batched_train_data = noisy_signals[key][i:i + window_size, 0]
        training_data.append(batched_train_data)
        i = i + window_size

    print(f"Peaks done and added to the dataset for record {key}")

    y_train = torch.Tensor(target_labels)
    X_train = torch.Tensor(training_data)
    trainData = Dataset(X_train, y_train)
    """
        X_val, y_val = torch.from_numpy(np.ravel(dataset[8])).float(), torch.from_numpy(top_labels[8]).float()
        valData = Dataset(X_val, y_val)
    
        X_test, y_test = torch.from_numpy(np.ravel(dataset[9])).float(), torch.from_numpy(top_labels[9]).float()
        testData = Dataset(X_test, y_test)"""

    # define training hyperparameters
    INIT_LR = 1e-3
    BATCH_SIZE = 10
    EPOCHS = 100

    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize the train, validation, and test data loaders
    trainDataLoader = DataLoader(trainData, shuffle=True,
                                 batch_size=BATCH_SIZE)
    """    valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
        testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)"""
    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    #valSteps = len(valDataLoader.dataset) // BATCH_SIZE

    print("Initializing model...")
    model = NoiseDetector(in_channels=1).to(device)
    opt = optim.Adam(model.parameters(), lr=INIT_LR)
    lossFn = nn.BCEWithLogitsLoss()

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


    # TRAINING
    # loop over epochs
    for e in tqdm(range(0, EPOCHS)):
        # train the model
        model.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        train_acc = 0
        val_acc = 0


        # loop over the training set
        for (x, y) in zip(X_train, y_train):
            x,y = torch.unsqueeze(x,0),torch.unsqueeze(y,0)
            x,y  = torch.unsqueeze(x,0), torch.unsqueeze(y,0)

            # break if end of dataset
            if x.shape[-1] < BATCH_SIZE:
                break

            # send the input to the device
            (x, y) = (x.to(device), y.to(device))

            opt.zero_grad()

            # perform a forward pass and calculate the training loss
            pred = model(x)


            #print(pred.shape, y.shape)
            loss = lossFn(pred, y)


            getPreds = nn.Sigmoid()
            predProb = getPreds(pred)

            train_acc = train_acc +  ((predProb>0.5) ==y) # correctly predicted samples

            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss = totalTrainLoss + loss



            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            loss.backward()
            opt.step()

        print(f"Epoch: {e+1}, Total training loss: {totalTrainLoss}")