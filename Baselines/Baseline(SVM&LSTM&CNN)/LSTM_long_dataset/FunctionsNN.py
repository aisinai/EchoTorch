import time, sys 


import numpy as np
import scipy.special
import math
import torch 
import matplotlib.pyplot as plt

__author__ = "Alessio Burrello"
__email__ = "alessio.burrello@studenti.polito.it"

def window_train_creation(EEG,fs, label): 
	''' Function used during training to take a bigger window of EEG and sum all the histograms
	of LGP frequency of 0.5 seconds windows inside.
	Parameters
		----------
	EEG: samples of EEG on multiple channels.
	fs: frequency of the signal.
	T: LGP length + 1
	label: label of the segment--> during test we can pass a random value, it will not be used
	totalNumberBP: dimensionality of the histogram
	Return  
	------
	quueryVector: we save all the histograms here creating a Vector for train of dimension 
	numberOfWindowsx(2^(T-1)xnChannels).
	'''		
	N_channels,learningEnd = EEG.shape
	index = np.arange(0,learningEnd-fs,fs/2)
	queeryVector = np.zeros(index.size*(fs*N_channels+1)).reshape(index.size,fs*N_channels+1)
	i = 0
	for iStep in index:
		queeryVector[i,0:fs*N_channels]= EEG[:,iStep:iStep+fs].reshape(N_channels*fs)
		queeryVector[i,fs*N_channels] = label
		i = i+1
	return queeryVector

def window_train_creation_spatial(EEG,fs, label): 
	''' Function used during training to take a bigger window of EEG and sum all the histograms
	of LGP frequency of 0.5 seconds windows inside.
	Parameters
		----------
	EEG: samples of EEG on multiple channels.
	fs: frequency of the signal.
	T: LGP length + 1
	label: label of the segment--> during test we can pass a random value, it will not be used
	totalNumberBP: dimensionality of the histogram
	Return  
	------
	quueryVector: we save all the histograms here creating a Vector for train of dimension 
	numberOfWindowsx(2^(T-1)xnChannels).
	'''		
	N_channels,learningEnd = EEG.shape
	index = np.arange(0,learningEnd,fs/2)
	X = np.zeros(index.size*fs/2*N_channels).reshape(index.size,fs/2,N_channels)
	y = np.zeros(index.size)
	i = 0
	for iStep in index:
		X[i,:,:]= EEG[:,iStep:iStep+fs/2].T
		y[i] = label
		i = i+1
	return X, y
