"""
This file converts the CinC 2017 Challenge Dataset into preferred format of 3 sec segments and reasssigns labels.
Possibly do some data augmentations etc.
"""
import wfdb
import wfdb.processing
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import time

class CinC_Formatter():
    def __init__(self):
        super(CinC_Formatter, self).__init__()

    def format(self, pathname, sampling_rate, window_size):
        self.pathname = pathname
        self.sampling_rate = sampling_rate
        self.window_size = window_size

        records_file = open(self.pathname)

        clean_signals = {}
        noisy_signals = {}

        name = ""
        for char in records_file.read():
            if char == '\n':
                continue
            name = name + char
            if len(name) == 6:
                clean_signals[name] = wfdb.rdrecord()

                name = ""
        print(clean_signals.keys())

        records_file.close()


if __name__ == "__main__":



