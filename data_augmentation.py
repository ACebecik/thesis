import matplotlib.pyplot as plt
import wfdb
import wfdb.processing
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# load the data from saved tensors
X_mit = torch.load("tensors/mit_all_records_X_w120_fixed.pt")
y_mit = torch.load("tensors/mit_all_records_y_w360.pt")


X_unovis = torch.load("tensors/unovis_all_records_X_w120.pt")
y_unovis = torch.load("tensors/unovis_all_records_y_w120.pt")

print(y_mit.shape)
print(y_unovis.shape)

X = np.vstack((X_mit, X_unovis))
y = np.concatenate((y_mit, y_unovis))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# Data Augmentation so that the class distribution is balanced
# 

print(f"Old cd for y_train:{sum(y_train)/y_train.shape[0] } ") 
init_train_length = y_train.shape[0]

for segment in tqdm(range(init_train_length//2)):
    if y_train[segment] == 1:
        scaler = MinMaxScaler(feature_range=(0,2))
        scaled_X_segment = scaler.fit_transform(X_train[segment, :].reshape(-1,1))
       # print(scaled_X_segment.shape)
       # print(X_train.shape, y_train.shape)

        X_train = np.vstack((X_train, scaled_X_segment.squeeze()))         
        y_train = np.hstack((y_train, np.array(1))) 

       # print(X_train.shape, y_train.shape)
        

print(f"New cd for y_train:{sum(y_train)/y_train.shape[0] } ")

torch.save(X_train, "tensors/augmented_UM_train_X.pt")
torch.save(y_train, "tensors/augmented UM_train_y.pt")

# test data augmentation


print(f"Old cd for y_test:{sum(y_test)/y_test.shape[0] } ") 
init_test_length = y_test.shape[0]

for segment in tqdm(range(init_test_length//2)):
    if y_test[segment] == 1:
        scaler = MinMaxScaler(feature_range=(0,2))
        scaled_X_segment = scaler.fit_transform(X_test[segment, :].reshape(-1,1))
       # print(scaled_X_segment.shape)
       # print(X_test.shape, y_test.shape)

        X_test = np.vstack((X_test, scaled_X_segment.squeeze()))         
        y_test = np.hstack((y_test, np.array(1))) 

       # print(X_test.shape, y_test.shape)
        

print(f"New cd for y_test:{sum(y_test)/y_test.shape[0] } ")

torch.save(X_test, "tensors/augmented_UM_test_X.pt")
torch.save(y_test, "tensors/augmented UM_test_y.pt")