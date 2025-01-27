import torch
import numpy as np
from tqdm import tqdm


# X=torch.load("tensors/mit_all_records_X_w360.pt")
# y=torch.load("tensors/mit_all_records_y_w360.pt")

X=torch.load("tensors/mit_clean_all_records_X_w360.pt")

print(X.shape)

X = X.cpu().numpy()
downsampled_X = np.zeros(120)
for i in tqdm (range (X.shape[0])):
    temp =  X[i,: :3]
    downsampled_X = np.vstack((downsampled_X, temp))

downsampled_X = np.delete(downsampled_X, (0),axis=0 ) 
print(downsampled_X.shape) 

downsampled_X_tensor = torch.Tensor(downsampled_X) 
torch.save(downsampled_X_tensor, "tensors/mit_clean_all_records_X_w120.pt")