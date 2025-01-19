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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix



class CustomDataset(Dataset):
    def __init__(self, x, y):
        # data loading
        self.x = x
        self.y = y

    def __getitem__(self, index):
        # allows for indexing
        get_x = self.x[index]
        get_y = self.y[index]
        return get_x,get_y

    def __len__(self):
        return len(self.x)


class NoiseDetector(nn.Module):
    def __init__(self, in_channels):
        super(NoiseDetector,self).__init__()

        # First set of Conv,Relu,Pooling,Dropout
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=0.3)

        # 2nd
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=0.3)

        #3rd
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(p=0.3)

        #4th
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(p=0.3)

        #FC layer and softmax
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(in_features=448, out_features=1024)
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

    """records_file = open("/media/medit-student/Volume/alperen/repo-clone/thesis/data/physionet.org/files/mitdb/1.0.0/RECORDS")
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



    y = torch.Tensor(target_labels)
    X = torch.Tensor(training_data)

   #save peaks for future 
    torch.save(X, "tensors/mit_all_records_X_w360.pt")
    torch.save(y, "tensors/mit_all_records_y_w360.pt")   


"""

    # define training hyperparameters
    INIT_LR = 1e-3
    BATCH_SIZE = 4096
    EPOCHS = 300

    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # load the data from saved tensors
    X = torch.load("tensors/unovis_all_records_X_w120.pt")
    y = torch.load("tensors/unovis_all_records_y_w120.pt")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=31)

    trainData = CustomDataset(X_train, y_train)
    testData = CustomDataset(X_test, y_test)

    # initialize the train, validation, and test data loaders
    trainDataLoader = DataLoader(trainData, shuffle=True,batch_size=BATCH_SIZE)
    testDataLoader = DataLoader(testData, shuffle=True, batch_size=BATCH_SIZE)
    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    testSteps = len(testDataLoader.dataset) // BATCH_SIZE

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

    results_train_acc = []
    results_train_loss = []
    results_val_acc = []
    results_val_loss = []

    conf_matrices_every_epoch =[] 


    # measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()


    # loop over epochs
    for e in tqdm(range(0, EPOCHS)):

        # TRAINING
        # train the model
        model.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        train_acc = 0
        val_acc = 0


        # loop over the training set in batches
        for X_batch, y_batch in trainDataLoader:
            X_batch,y_batch = torch.unsqueeze(X_batch,1),torch.unsqueeze(y_batch,1)
            #X_batch,y_batch = torch.unsqueeze(X_batch,0), torch.unsqueeze(y_batch,0)

            """            # break if end of dataset
                        if X_batch.shape[-1] < BATCH_SIZE:
                            break"""

            # send the input to the device
            (X_batch, y_batch) = (X_batch.to(device), y_batch.to(device))

            opt.zero_grad()
            # perform a forward pass and calculate the training loss
            pred = model(X_batch)
            #print(pred.shape, y.shape)
            loss = lossFn(pred, y_batch)
            getPreds = nn.Sigmoid()
            predProb = getPreds(pred)

            train_acc = train_acc +  ((predProb>0.5) ==y_batch).sum() # correctly predicted samples

            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss = totalTrainLoss + loss


            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            loss.backward()
            opt.step()

        avgTrainAcc = float(train_acc/len(y_train))
        avgTrainLoss = float(totalTrainLoss /trainSteps)
        if e % 10 == 0:
            print(str.format("Epoch: {}, Avg training loss: {:.6f}, Avg Train Acc: {:.6f}", e+1, avgTrainLoss, avgTrainAcc))

        # update our training history
        results_train_acc.append(avgTrainAcc)
        results_train_loss.append(avgTrainLoss)


        #EVAL
        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()

            # clear confusion matrix
            conf_matrix =np.zeros((2,2)) 
 
            # loop over the validation set
            for X_batch, y_batch in testDataLoader:
                X_batch, y_batch  = torch.unsqueeze(X_batch,1), torch.unsqueeze(y_batch,1)

                """                # break if end of dataset
                                if x.shape[-1] < BATCH_SIZE:
                                    break"""

                # send the input to the device
                (X_batch, y_batch) = (X_batch.to(device), y_batch.to(device))

                # make the predictions and calculate the validation loss
                pred = model(X_batch)
                lossVal = lossFn(pred, y_batch)

                predProbVal = getPreds(pred)

                totalValLoss = totalValLoss + lossVal

                val_acc = val_acc + ((predProbVal>0.5)==y_batch).sum()

               # confusion matrix 
                predictions = (predProbVal>0.5)*1
                temp_conf_matrix = confusion_matrix(y_batch.cpu().numpy(), predictions.cpu().numpy())
                conf_matrix = np.add(conf_matrix, temp_conf_matrix)

        avgValAcc = float(val_acc/len(y_test))
        avgValLoss = float(totalValLoss /testSteps)
        if e % 10 == 0:
            print(str.format("Epoch: {}, Avg Validation loss: {:.6f}, Avg Val Acc: {:.6f}", e+1, avgValLoss, avgValAcc))

        # update our training history
        results_val_acc.append(avgValAcc)
        results_val_loss.append(avgValLoss)
        conf_matrices_every_epoch.append(conf_matrix)

    # finish measuring how long training took
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))


    plt.plot(results_train_acc, label='Train Acc')
    plt.plot(results_val_acc, label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"plots/unovis_all_Acc_plot_w120_lr{INIT_LR}_batchsize{BATCH_SIZE}.png")
    plt.clf()

    plt.plot(results_train_loss, label='Train Loss')
    plt.plot(results_val_loss, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.title('Loss')
    plt.legend()
    plt.savefig(f"plots/unovis_all_Loss_plot_w120_lr{INIT_LR}_batchsize{BATCH_SIZE}.png")
    plt.clf()

   #display confusion matrix for the best accuracy epoch
    best_epoch = np.argmax(results_val_acc)
    disp_conf_matrix = ConfusionMatrixDisplay(conf_matrices_every_epoch[best_epoch])
    disp_conf_matrix.plot()
    plt.savefig(f"plots/unovis_all_Conf_Matrix_w120_lr{INIT_LR}_batchsize{BATCH_SIZE}.png")
    plt.clf()