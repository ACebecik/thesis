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
from tqdm import tqdm
import wfdb
import wfdb.processing
import scipy
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.metrics import classification_report



class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index])
        y = torch.Tensor(self.y[index])
        return (x, y)

    def __len__(self):
        count = self.x.shape[0]
        return count


class NoiseDetector(nn.Module):
    def __init__(self, in_channels):
        super(NoiseDetector,self).__init__()

        # First set of Conv,Relu,Pooling,Dropout
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=7, stride=1)
        self.relu1 = nn.ReLU()
        #self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=0.7)

        # 2nd
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1)
        self.relu2 = nn.ReLU()
        #self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=0.7)

        #3rd
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1)
        self.relu3 = nn.ReLU()
        #self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(p=0.7)

        #4th
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=1)
        self.relu4 = nn.ReLU()
        #self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(p=0.7)

        #FC layer and softmax
        #self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(in_features=40, out_features=1024)
        self.relu5 = nn.ReLU()

        #self.flatten2 = nn.Flatten()
        self.fc2 = nn.Linear(in_features=1024, out_features=2)
        self.logSoftmax = nn.LogSoftmax()


    def forward(self, x):
        "Define the forward pass"

        #print('before conv1 layer x.shape:', x.shape)

        #layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        #x = self.pool1(x)
        x = self.drop1(x)

        #print('before conv2 layer x.shape:', x.shape)

        #layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        #x = self.pool2(x)
        x = self.drop2(x)

        #print('before conv3 layer x.shape:', x.shape)

        #layer 3
        x = self.conv3(x)
        x = self.relu3(x)
        #x = self.pool3(x)
        x = self.drop3(x)

        #print('before conv4 layer x.shape:', x.shape)

        #layer 4
        x = self.conv4(x)
        x = self.relu4(x)
        #x = self.pool4(x)
        x = self.drop4(x)

        #print('before linear1 layer x.shape:', x.shape)
        #x = self.flatten1(x)
        x = self.fc1(x)
        x = self.relu5(x)

        #print('before linear2 layer x.shape:', x.shape)

        #x = self.flatten2(x)
        x = self.fc2(x)
        #print('after linear2 layer x.shape:', x.shape)
        output = self.logSoftmax(x)

        return output


if __name__ == "__main__":

    records_file = open("/Users/alperencebecik/Desktop/Thesis Masterfile/data/mit-bih-arrhythmia-database-1.0.0/RECORDS")
    noise_em = wfdb.rdrecord("/Users/alperencebecik/Desktop/Thesis Masterfile/data/mit-bih-noise-stress-test-database-1.0.0/em").p_signal

    clean_signals = {}
    noisy_signals = {}
    noise_perc = 0.2


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
    window_size = 3 * fs
    dataset = []
    top_labels = []

    # find if peaks match = usable, if not = unusable
    for key in clean_signals.keys():

        record_length = len(noisy_signals[key])
        labels = np.zeros((record_length, 2))  # 0: not usable class , 1: usable ecg class
        i = 0
        while i < record_length:
            if i + window_size >= record_length:
                rpeaks = wfdb.processing.xqrs_detect(clean_signals[key][i:, 0], fs=360, verbose=False)
                rpeaks_noisy = wfdb.processing.xqrs_detect(noisy_signals[key][i:, 0], fs=360, verbose=False)
                if set(rpeaks) == set(rpeaks_noisy):
                    labels[i:, 1] = 1
                else:
                    labels[i:, 0] = 1

                break
            else:
                rpeaks = wfdb.processing.xqrs_detect(clean_signals[key][i:i+window_size, 0], fs=360, verbose=False)
                rpeaks_noisy = wfdb.processing.xqrs_detect(noisy_signals[key][i:i+window_size, 0], fs=360, verbose=False)
                if set(rpeaks) == set(rpeaks_noisy):
                    labels[i:i+window_size, 1] = 1
                else:
                    labels[i:i+window_size, 0] = 1

            i = i + window_size

        dataset.append(noisy_signals[key][:,0])
        top_labels.append(labels)
        print(f"Peaks done and added to the dataset for record {key}")

        if len(dataset) == 10:
            break

    y_train = torch.from_numpy(np.reshape(np.ravel(top_labels[:8]), (len(np.ravel(top_labels[:8]))//2,2))).float()
    X_train = torch.from_numpy(np.ravel(dataset[:8])).float()
    trainData = Dataset(X_train, y_train)

    X_val, y_val = torch.from_numpy(np.ravel(dataset[8])).float(), torch.from_numpy(top_labels[8]).float()
    valData = Dataset(X_val, y_val)

    X_test, y_test = torch.from_numpy(np.ravel(dataset[9])).float(), torch.from_numpy(top_labels[9]).float()
    testData = Dataset(X_test, y_test)

    # define training hyperparameters
    INIT_LR = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 10
    # define the train and val splits
    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 1 - TRAIN_SPLIT
    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize the train, validation, and test data loaders
    trainDataLoader = DataLoader(trainData, shuffle=False,
                                 batch_size=BATCH_SIZE)
    valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
    testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)
    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // BATCH_SIZE

    print("Initializing model...")
    model = NoiseDetector(in_channels=1).to(device)
    opt = optim.Adam(model.parameters(), lr=INIT_LR)
    lossFn = nn.CrossEntropyLoss()

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


    """
        # loop over epochs
        for e in tqdm(range(0, EPOCHS)):
            # train the model
            model.train()
    
            # initialize the total training and validation loss
            totalTrainLoss = 0
            totalValLoss = 0
            # initialize the number of correct predictions in the training
            # and validation step
            trainCorrect = 0
            valCorrect = 0
            # loop over the training set
            for (x, y) in trainDataLoader:
                x = torch.unsqueeze(x,0)
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))
    
                # perform a forward pass and calculate the training loss
                pred = model(x)
                #print(pred.shape, y.shape)
                loss = lossFn(pred, y)
                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far and
                # calculate the number of correct predictions
                totalTrainLoss += loss
                trainCorrect += (pred.argmax(1) == y.argmax(1)).type(
                    torch.float).sum().item()
    
    """

    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    valCorrect = 0

    # switch off autograd for evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x, y) in valDataLoader:
            x = torch.unsqueeze(x,0)

            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # make the predictions and calculate the validation loss
            pred = model(x)
            totalValLoss += lossFn(pred, y)
            # calculate the number of correct predictions
            valCorrect += (pred.argmax(1) == y.argmax(1)).type(
                torch.float).sum().item()

    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    # calculate the training and validation accuracy
    trainCorrect = trainCorrect / len(trainDataLoader.dataset)
    valCorrect = valCorrect / len(valDataLoader.dataset)
    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
        avgValLoss, valCorrect))

    # finish measuring how long training took
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    """    
        # we can now evaluate the network on the test set
        print("[INFO] evaluating network...")
        # turn off autograd for testing evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
    
            # initialize a list to store our predictions
            preds = []
            # loop over the test set
            for (x, y) in testDataLoader:
                x = torch.unsqueeze(x,0)
    
                # send the input to the device
                x = x.to(device)
                # make the predictions and add them to the list
                pred = model(x)
                preds.extend(pred.argmax(axis=1).cpu().numpy())
        # generate a classification report
        print(classification_report(testData.targets.cpu().numpy(),
                                    np.array(preds), target_names=testData.classes))"""