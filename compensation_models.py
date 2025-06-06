"""This file implements the compensation models for motion artifacts. 
Implemented models:
1- DRDNN method from Antczak, 2018
2- Fully Conv Net DAE from Chiang, 2019"""


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
from custom_dataset_for_dataloader import CustomDataset
from classification_models import NoiseDetector
from OLD_load_data_from_tensors import LoadDataFromTensor
from train import ClassificationTrainer
from plotter import Plotter


class DRDNN(nn.Module):
    def __init__(self, lstm_hidden_size):
        super(DRDNN,self).__init__()

        self.lstm = nn.LSTM(input_size=120, hidden_size=lstm_hidden_size)
        self.fc1 = nn.Linear(in_features=lstm_hidden_size, out_features=lstm_hidden_size)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=lstm_hidden_size, out_features=lstm_hidden_size)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(in_features=lstm_hidden_size, out_features=120)

    def forward(self, x):

        out, (ht, ct) = self.lstm(x)
        out = out[:, -1, :]

        x = self.fc1(out)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x
    


class FCN_DAE(nn.Module):
    def __init__(self, in_channels=1, p_dropout=0.1):
        super(FCN_DAE, self).__init__()

        # First set of Conv,Relu,Pooling,Dropout
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=20, kernel_size=16, stride=2, padding=7)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=p_dropout)

        # 2nd
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=40, kernel_size=16, stride=2, padding=7)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=p_dropout)

        #3rd
        self.conv3 = nn.Conv1d(in_channels=40, out_channels=80, kernel_size=16, stride=2, padding=7)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=p_dropout)

        #4th
        self.conv4 = nn.Conv1d(in_channels=80, out_channels=160, kernel_size=8, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=p_dropout)

        # decoder

        self.deconv1 = nn.ConvTranspose1d(in_channels=160, out_channels=80, kernel_size=8, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose1d(in_channels=80, out_channels=40, kernel_size=16, stride=2, padding=7)
        self.relu6 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose1d(in_channels=40, out_channels=20, kernel_size=16, stride=2, padding=7)
        self.relu7 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose1d(in_channels=20, out_channels=1, kernel_size=16, stride=2, padding=7)

    def forward(self, x):
        # Defines the forward pass

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.drop4(x)

        x = self.deconv1(x)
        x = self.relu5(x)

        x = self.deconv2(x)
        x = self.relu6(x)

        x = self.deconv3(x)
        x = self.relu7(x)

        x = self.deconv4(x)

        return x



class FCN_DAE_skip(nn.Module):
    def __init__(self, in_channels=1):
        super(FCN_DAE_skip, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=20, kernel_size=16, stride=2, padding=7)
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=40, kernel_size=16, stride=2, padding=7)
        self.conv3 = nn.Conv1d(in_channels=40, out_channels=80, kernel_size=16, stride=2, padding=7)
        self.conv4 = nn.Conv1d(in_channels=80, out_channels=160, kernel_size=8, stride=1, padding=1)

        self.relu = nn.ReLU()
       # self.batchnorm = nn.BatchNorm1d()

        self.deconv1 = nn.ConvTranspose1d(in_channels=160, out_channels=80, kernel_size=8, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels=80, out_channels=40, kernel_size=16, stride=2, padding=7)
        self.deconv3 = nn.ConvTranspose1d(in_channels=40, out_channels=20, kernel_size=16, stride=2, padding=7)
        self.deconv4 = nn.ConvTranspose1d(in_channels=20, out_channels=1, kernel_size=16, stride=2, padding=7)
    
        self.maxpool20 = nn.MaxPool1d(kernel_size=20)
        self.maxpool40 = nn.MaxPool1d(kernel_size=40)
        self.maxpool80 = nn.MaxPool1d(kernel_size=80)

        self.avgpool20 = nn.AvgPool1d(kernel_size=20)
        self.avgpool40 = nn.AvgPool1d(kernel_size=40)
        self.avgpool80 = nn.AvgPool1d(kernel_size=80)

        self.adp_maxpool = nn.AdaptiveMaxPool1d(output_size=1).to(torch.device("cuda:0"))
        self.adp_avgpool = nn.AdaptiveAvgPool1d(output_size=1).to(torch.device("cuda:0"))

        self.adp_avgpool_chn = nn.AdaptiveAvgPool1d(output_size=1).to(torch.device("cuda:0"))
        self.max_avgpool_chn = nn.AdaptiveMaxPool1d(output_size=1).to(torch.device("cuda:0"))

        self.linear20_avg = nn.Linear(in_features=1, out_features=20).to(torch.device("cuda:0"))
        self.linear20_max = nn.Linear(in_features=1, out_features=20).to(torch.device("cuda:0"))
        self.linear20_avg_rev = nn.Linear(in_features=20, out_features=1).to(torch.device("cuda:0"))
        self.linear20_max_rev = nn.Linear(in_features=20, out_features=1).to(torch.device("cuda:0"))


        self.linear40_avg = nn.Linear(in_features=1, out_features=40).to(torch.device("cuda:0"))
        self.linear40_max = nn.Linear(in_features=1, out_features=40).to(torch.device("cuda:0"))
        self.linear40_avg_rev = nn.Linear(in_features=40, out_features=1).to(torch.device("cuda:0"))
        self.linear40_max_rev = nn.Linear(in_features=40, out_features=1).to(torch.device("cuda:0"))

        self.linear80_avg = nn.Linear(in_features=1, out_features=80).to(torch.device("cuda:0"))
        self.linear80_max = nn.Linear(in_features=1, out_features=80).to(torch.device("cuda:0"))
        self.linear80_avg_rev = nn.Linear(in_features=80, out_features=1).to(torch.device("cuda:0"))
        self.linear80_max_rev = nn.Linear(in_features=80, out_features=1).to(torch.device("cuda:0"))


        self.att_conv20 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1).to(torch.device("cuda:0"))
        self.att_conv40 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1).to(torch.device("cuda:0"))
        self.att_conv80 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1).to(torch.device("cuda:0"))


        
    def dual_attention(self, x, layer_no):
        
        channel_attention = self.global_attention(x,layer_no=layer_no)
        spatial_attention = self.spatial_attention(x, layer_no=layer_no)
        temp = (x * spatial_attention* channel_attention)
        return temp

    def global_attention(self, x, layer_no):
    
       # global max and avg poolings per the channels
        """
                x_avgpooled = self.adp_avgpool(x.permute(0,2,1))
                x_maxpooled = self.adp_maxpool(x.permute(0,2,1))
"""

        x_avgpooled = self.adp_avgpool_chn(x)
        x_maxpooled = self.max_avgpool_chn(x)

        if layer_no == 20:
        
            x_avgpooled = self.linear20_avg(x_avgpooled)
            x_maxpooled = self.linear20_max(x_maxpooled)

            x_avgpooled = self.relu(x_avgpooled)
            x_maxpooled = self.relu(x_maxpooled)

            x_avgpooled = self.linear20_avg_rev(x_avgpooled)
            x_maxpooled = self.linear20_max_rev(x_maxpooled)
        
        elif layer_no == 40:

            x_avgpooled = self.linear40_avg(x_avgpooled)
            x_maxpooled = self.linear40_max(x_maxpooled)
            
            x_avgpooled = self.relu(x_avgpooled)
            x_maxpooled = self.relu(x_maxpooled)

            x_avgpooled = self.linear40_avg_rev(x_avgpooled)
            x_maxpooled = self.linear40_max_rev(x_maxpooled)
        
        else:
            x_avgpooled = self.linear80_avg(x_avgpooled)
            x_maxpooled = self.linear80_max(x_maxpooled)
        
            x_avgpooled = self.relu(x_avgpooled)
            x_maxpooled = self.relu(x_maxpooled)

            x_avgpooled = self.linear80_avg_rev(x_avgpooled)
            x_maxpooled = self.linear80_max_rev(x_maxpooled)

        x = x_avgpooled + x_maxpooled
        sigmoid1 = nn.Sigmoid().to(torch.device("cuda:0"))
        channel_attention = sigmoid1(x)

       # outputs in the same shape of x  
        return channel_attention

    def spatial_attention(self,x,layer_no):
        
        """        x_avgpooled = self.adp_avgpool_chn(x)
                x_maxpooled = self.max_avgpool_chn(x)
        """

        x_avgpooled = self.adp_avgpool(x.permute(0,2,1))
        x_maxpooled = self.adp_maxpool(x.permute(0,2,1))

       # stack along features 
        temp = torch.stack((x_avgpooled, x_maxpooled), dim=1)
        temp = torch.squeeze(temp)
        

        if layer_no == 20:
            temp = self.att_conv20(temp)

        elif layer_no == 40 :
            temp = self.att_conv40(temp)
        
        else:
            temp = self.att_conv80(temp)

        sigm = nn.Sigmoid().to(torch.device("cuda:0"))
        temp = sigm(temp)

        return temp

    def forward(self,x):

        x1 = self.conv1(x)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.relu(x3)

        x4 = self.conv4(x3)
        x4 = self.relu(x4)

        x4 = self.deconv1(x4)
        x5 = self.relu(x4)

       # dual attention
        x5 = self.dual_attention(x5, 80) 

       # skip connect 
        x3 = x5 + x3 
        x3 = self.deconv2(x3)
        x6 = self.relu(x3)

        x6 = self.dual_attention(x6, 40)

       # skip connect 
        x2 = x2 + x6
        x2 = self.deconv3(x2)
        x7 = self.relu(x2)

        x7 = self.dual_attention(x7, 20)

       # skip connect 
        x1 = x1 + x7
        x = self.deconv4(x1)

        return x

if __name__ == "__main__":
    model_temp = FCN_DAE_skip().to(device="cuda:0")
    X = torch.Tensor (np.ones((1024,1,120))).to(device="cuda:0")
    model_temp.forward(X)

    
    