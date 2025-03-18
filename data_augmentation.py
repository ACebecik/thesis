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
um_train_X = torch.load("tensors/final_tensors_1703/um_train_X.pt")
um_train_y = torch.load("tensors/final_tensors_1703/um_train_y.pt")
um_reference_train_X = torch.load("tensors/final_tensors_1703/um_reference_train_X.pt")


# Data Augmentation

print(f"Old cd for y_train:{sum(um_train_y)/um_train_y.shape[0] } ") 
init_train_length = um_train_y.shape[0]

augmented_train_segments_x =[]
class_labels_y = []
reference_train_segments_x =[] 
scaler = MinMaxScaler(feature_range=(0,2))


for segment in tqdm(range(init_train_length)):

    scaled_X_segment = scaler.fit_transform(um_train_X[segment, :].reshape(-1,1)).squeeze()

   #add newly scaled segment to the train set  
    augmented_train_segments_x.append(scaled_X_segment)
    class_labels_y.append(um_train_y[segment])
    reference_train_segments_x.append(um_reference_train_X[segment,:] )

augmented_train_segments_x = np.array(augmented_train_segments_x, dtype=np.float32)
class_labels_y = np.array(class_labels_y, dtype=np.float32)
reference_train_segments_x = np.array(reference_train_segments_x, dtype=np.float32)

#add newly scaled segment to the train set  
aum_train_X = np.vstack((um_train_X, augmented_train_segments_x))
aum_train_y = np.hstack((um_train_y, class_labels_y)) 
aum_reference_train_X = np.vstack((um_reference_train_X, reference_train_segments_x))         


torch.save(torch.Tensor(aum_train_X), "tensors/final_tensors_1703/augmented_um_train_X.pt")
torch.save(torch.Tensor(aum_train_y), "tensors/final_tensors_1703/augmented um_train_y.pt")
torch.save(torch.Tensor(aum_reference_train_X), "tensors/final_tensors_1703/augmented um_reference_train_X.pt")