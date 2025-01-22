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
from load_data_from_tensors import LoadDataFromTensor
from train import ClassificationTrainer


class Plotter():
    def __init__(self, dataset, seed, lr, batch_size, save_path = "plots"):
        
        self.dataset = dataset
        self.seed = seed
        self.lr = lr
        self.batch_size = batch_size
        self.save_path = save_path

    
    def plot_accuracy(self, train_acc, test_acc):

        plt.plot(train_acc, label='Train Acc')
        plt.plot(test_acc, label='Val Acc')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{self.save_path}/{self.dataset}_seed_{self.seed}_lr_{self.lr}_bs_{self.batch_size}_ACC.png")
        plt.clf()

    def plot_loss(self, train_loss, test_loss):

        plt.plot(train_loss, label='Train Loss')
        plt.plot(test_loss, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel("Loss")
        plt.title('Loss')
        plt.legend()
        plt.savefig(f"{self.save_path}/{self.dataset}_seed_{self.seed}_lr_{self.lr}_bs_{self.batch_size}_LOSS.png")
        plt.clf()

    def plot_confusion_matrix(self, conf_matrix):

        disp_conf_matrix = ConfusionMatrixDisplay(conf_matrix)
        disp_conf_matrix.plot()
        plt.savefig(f"{self.save_path}/{self.dataset}_seed_{self.seed}_lr_{self.lr}_bs_{self.batch_size}_CONF.png")
        plt.clf()
    