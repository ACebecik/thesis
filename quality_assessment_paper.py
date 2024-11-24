"""
This file implements the quality assessment paper:
Noise Detection in Electrocardiography Signal for Robust Heart Rate Variability Analysis: A Deep Learning Approach
by Ansari et al.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import wfdb
import wfdb.processing

import scipy



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

    #toy_dataset = scipy.datasets.electrocardiogram()
    rpeaks = wfdb.processing.xqrs_detect(records['100'].p_signal[:,0], fs=360, verbose=False)
    plt.plot(records['100'].p_signal[:,0])
    plt.plot(rpeaks, records['100'].p_signal[:,0][rpeaks], "x")
    plt.show()