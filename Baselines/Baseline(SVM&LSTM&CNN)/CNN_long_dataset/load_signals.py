import os
import numpy as np
import pandas as pd
import scipy.io as scp
from scipy.signal import resample
from scipy.signal import stft
import matplotlib.pyplot as plt


class PrepData():
    def __init__(self, EEG, frequency,segment):
        self.EEG = EEG
        self.frequency = frequency
        self.segment = segment


    def preprocess_train(self):
        ictal_nInterictal = self.segment == 'ictal'
        N_channels,learningEnd = self.EEG.shape       
        # w_len = 2 seconds
        window_dim = self.frequency*9
        index = np.arange(0,learningEnd-window_dim,self.frequency)
        i = 0
        X =[]
        y = []
        for iStep in index:
            f,t,stft_data = stft(self.EEG[:,iStep:iStep+window_dim],fs = 512,nperseg=256)
            stft_data = stft_data[:,0:75,:]
            stft_data = np.transpose(stft_data,(0,2,1))
            stft_data = np.abs(stft_data)+1e-6
            stft_data = np.log10(stft_data)
            indices = np.where(stft_data <= 0)
            stft_data[indices] = 0
            stft_data = stft_data.reshape(-1, 1, stft_data.shape[0],
                                                stft_data.shape[1],
                                                stft_data.shape[2])
            X.append(stft_data)
            if ictal_nInterictal:
                y.append(1)
            else:
                y.append(0)
        X=np.concatenate(X,axis=0)
        y=np.array(y)
        return X, y


    def apply(self):
        X, y = self.preprocess_train()      
        return X,y





















