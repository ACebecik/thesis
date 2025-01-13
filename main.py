import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import torchvision
# import torchvision.transforms as transforms
#from torchvision.datasets import ImageFolder
from tqdm import tqdm

import scipy.io
import matplotlib.pyplot as plt
import numpy as np

from drdnn import DRRN_denoising

class LoadDatasets():
    def __init__(self, num_participants, data):
        super(LoadDatasets, self).__init__()

        for i in range(1, num_participants+1):
            data[i] = scipy.io.loadmat(f'/Users/alperencebecik/Desktop/Thesis Masterfile/Unovis_cumumovi2023/participant{i}.mat')

    def train_test_split(self, test_participants, data):

        train_data = dict(list(data.items())[test_participants:len(data.keys())])
        test_data = dict(list(data.items())[:test_participants])

        return train_data, test_data

if __name__ == '__main__':


    data = {}
    train_data , test_data = LoadDatasets(10, data).train_test_split(2, data)


    train_data_loader = DataLoader(list(zip(train_data[4]['ecg1'],train_data[4]['ecg_ref'])), batch_size=64, shuffle=False)
    test_data_loader = DataLoader(list(zip(test_data[1]['ecg1'],test_data[1]['ecg_ref'])), batch_size=64, shuffle=False)


    """
        for X_batch, y_batch in tqdm(train_data_loader, desc='Training loop'):
            # Move inputs and labels to the device
            X_batch, y_batch = X_batch.to(torch.float32), y_batch.to(torch.float32)
            X_batch = torch.unsqueeze(X_batch, 1).reshape(1,1,64)
            y_batch = torch.unsqueeze(y_batch, 1)
            print(X_batch.shape)"""

    model = DRRN_denoising()
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_losses = []
    test_losses = []
    num_epochs = 10

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in tqdm(train_data_loader, desc='Training loop'):
        # Move inputs and labels to the device
        X_batch, y_batch = X_batch.to(torch.float32), y_batch.to(torch.float32)
        X_batch = torch.unsqueeze(X_batch, 1).reshape(1,1,64)
        y_batch = torch.unsqueeze(y_batch, 1).reshape(1,1,64)


        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y_batch.size(0)
    train_loss = running_loss / len(train_data_loader.dataset)
    train_losses.append(train_loss)

    # Test phase
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_data_loader, desc='Validation loop'):
            # Move inputs and labels to the device
            X_batch, y_batch = X_batch.to(torch.float32), y_batch.to(torch.float32)

            outputs = model(X_batch)
            loss = loss(outputs, y_batch)
            running_loss += loss.item() * y_batch.size(0)
    val_loss = running_loss / len(test_data_loader.dataset)
    test_losses.append(val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")


"""    mat = scipy.io.loadmat('/Users/alperencebecik/Desktop/Thesis Masterfile/Unovis_cumumovi2023/participant3.mat')
    #mat = scipy.io.loadmat('/Users/alperencebecik/Desktop/Thesis Masterfile/UnoViS_auto2012.mat')

    ecg1 = mat["ecg1"]
    ecg2 = mat["ecg2"]
    ecg3 = mat["ecg3"]
    ecg4 = mat["ecg4"]
    ecg_ref = mat["ecg_ref"]

    f1 = plt.figure(1)
    plt.plot(ecg_ref[120000:], label='ref')
    plt.plot(ecg1[120000:], label="raw")
    #plt.plot(ref_data[1000000:1500000], label='ref')
    plt.legend()
    plt.show()

    train_dataloader = DataLoader(ecg1, batch_size=64, shuffle=False)
    test_dataloader ="""