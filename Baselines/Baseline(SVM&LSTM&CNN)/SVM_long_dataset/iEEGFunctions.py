#!/usr/bin/env python

''' 
Local sensitive Hashing function 
'''
import time, sys 


import numpy as np
import scipy.special
import math
import torch 
import matplotlib.pyplot as plt

__author__ = "Alessio Burrello"
__email__ = "alessio.burrello@studenti.polito.it"
class HD_classifier:

	def __init__(self, N_seats1,HD_dim, N_seats2, T, device, string, cuda = True):
		''' Initialize an HD classifier using the torch library 
		Parameters
		----------
		N_seats1: number of elements inside the LBP item memory
		HD_dim: dimension of the HD vectprs
		N_seats2: number of channels of the channels item memory
		T: LBP length + 1
		device: gpu to be used to create the itemMemory
		string: type of item Memory for the channel item Memory: random or sandwich
		cuda: this paramether is fixed to true by now. The code MUST be ran on GPU.
		'''	
		self.training_done = False 
		self.N_seats1 = N_seats1
		self.N_seats2 = N_seats2
		self.HD_dim = HD_dim
		self.T = T	
		self.device = device	
		# creation of a random itemMemory of 1 and 0 of dimension N_seats1xHD_dim	
		if cuda:
			self.proj_mat_LBP= torch.randn(self.HD_dim, self.N_seats1).cuda(device = device)
		else:
			self.proj_mat_LBP= torch.randn(self.HD_dim, self.N_seats1)
		self.proj_mat_LBP[self.proj_mat_LBP >0] = 1
		self.proj_mat_LBP[self.proj_mat_LBP <= 0] = 0
		# creation of a sandwich itemMemory of 1 and 0 of dimension N_seats2xHD_dim: the single
		#hypervectors inside the memory are similar to adiacent ones (~75% equal) and pseudo orthogonal
		#with the others.
		if string == 'sandwich':
			if cuda:
				self.proj_mat_channels= torch.zeros(self.HD_dim, self.N_seats2).cuda(device = device)
			else:
				self.proj_mat_channels= torch.zeros(self.HD_dim, self.N_seats2)
			for i in range(N_seats2):
				if i%2 == 0:
					self.proj_mat_channels[:,i]= torch.randn(self.HD_dim, 1).cuda(device = device)
					self.proj_mat_channels[self.proj_mat_channels >0] = 1
					self.proj_mat_channels[self.proj_mat_channels <= 0] = 0					
			for i in range(N_seats2-1):
				if i%2 == 1:
					self.proj_mat_channels[0:HD_dim/2,i] = self.proj_mat_channels[0:HD_dim/2,i-1]
					self.proj_mat_channels[HD_dim/2:HD_dim,i] = self.proj_mat_channels[HD_dim/2:HD_dim,i+1]
			self.proj_mat_channels[0:HD_dim/2,N_seats2-1] = self.proj_mat_channels[0:HD_dim/2,N_seats2-2]
			if cuda:
				self.proj_mat_channels[HD_dim/2:HD_dim,N_seats2-1] = torch.randn(self.HD_dim/2, 1).cuda(device = device)
			else:
				self.proj_mat_channels[HD_dim/2:HD_dim,N_seats2-1] = torch.randn(self.HD_dim/2, 1)
			self.proj_mat_channels[self.proj_mat_channels >0] = 1
			self.proj_mat_channels[self.proj_mat_channels <= 0] = 0	
		# creation of a random itemMemory of 1 and 0 of dimension N_seats2xHD_dim
		elif string == 'random':
			if cuda:
				self.proj_mat_channels= torch.randn(self.HD_dim, self.N_seats2).cuda(device = device)
			else:
				self.proj_mat_channels= torch.randn(self.HD_dim, self.N_seats2)
			self.proj_mat_channels[self.proj_mat_channels >=0] = 1
			self.proj_mat_channels[self.proj_mat_channels < 0] = 0	


	def learn_HD_proj(self,EEG): 
		''' starting from EEG data, it creates the histogram of LBP for the target segment
		projecting it in the HD space.
		Parameters
		----------
		EEG: samples of EEG on multiple channels.
		Return  
		------
		quueryVector: HV in which all the histograms among all the channels are encoded.
		'''	
		queeryVector = torch.cuda.ShortTensor(1,1).zero_()
		N_channels,learningEnd = EEG.size()
		LBP_weights = torch.cuda.ShortTensor([2**0, 2**1, 2**2, 2**3, 2**4, 2**5])
		for iStep in range(learningEnd-6):
			x = EEG[:,iStep:(iStep+self.T)].short()
			bp = (torch.add(-x[:,0:self.T-1], 1,x[:,1:self.T])>0).short()
			value = torch.sum(torch.mul(LBP_weights,bp), dim=1)
			bindingVector=self.xor(self.proj_mat_channels,self.proj_mat_LBP[:,value.long()])
			output_vector=torch.sum(bindingVector,dim=1).short()
			#here we broke ties summing an additional HV in case of an even number
			if N_channels%2==0:
				output_vector = torch.add(self.xor(self.proj_mat_LBP[:,1],self.proj_mat_LBP[:,2]),1,output_vector)
			output_vector=(output_vector>int(math.floor(self.N_seats2/2))).short()
			queeryVector=torch.add(queeryVector,1,output_vector)
		queeryVector = (queeryVector> (learningEnd-6)/2).short()
		return queeryVector

		
	def learn_HD_proj_big_half(self,EEG,fs, window_dim): 
		''' Function used during training to take a bigger window of EEG and sum all the histograms
		of LBP frequency of 0.5 seconds windows inside.
		Parameters
		----------
		EEG: samples of EEG on multiple channels.
		fs: frequency of the signal.
		Return  
		------
		quueryVector: HV in which all the informations of this big window of EEG is encoded.
		'''			
		queeryVector = torch.cuda.ShortTensor(1,1).zero_()
		N_channels,learningEnd = EEG.size()
		index = np.arange(0,learningEnd-window_dim,window_dim)
		for iStep in index:
			temp = self.learn_HD_proj(EEG[:,iStep:iStep+window_dim])
			queeryVector=torch.add(queeryVector,1,temp)
		queeryVector = (queeryVector > index.size/2).short()
		return queeryVector

	def predict(self,testVector,Ictalprot, Interictalprot,D): 
		''' Prediction function of HD: it gives in addition to the class prediction also the
		distance from the 2 class prototypes.
		Parameters
		----------
		testVector: HV of the unlabled segment.
		Ictalprot: prototype (HD vector) for the ictal state.
		Interictalprot: prototype (HD vector) for the interictal state.
		Return  
		------
		distanceVectorsS: distance from ictal Prototype
		distanceVectornS: distance from interictal prototype
		prediction: 1 for seizure, 0 for interictal.
		'''				
		distanceVectorsS = self.ham_dist(testVector,Ictalprot,D)
		distanceVectorsnS = self.ham_dist(testVector,Interictalprot,D)
		if distanceVectorsS > distanceVectorsnS:
			prediction = 0
		else:
			prediction = 1
		return distanceVectorsS,distanceVectorsnS,prediction
	
	def xor(self,vec_a,vec_b):
		''' xor between vec_a and vec_b
		Parameters
		----------
		vec_a: first vector, torch Short Tensor shape (HD_dim,)
		vec_b: second vector, torch Short Tensor shape (HD_dim,)
		Return  
		------
		vec_c: vec_a xor vec_b
		'''	
		vec_c = (torch.add(vec_a,vec_b) == 1).short()  # xor  

		return vec_c

	def ham_dist(self,vec_a,vec_b,D):
		''' calculate relative hamming distance 
		Parameters
		----------
		vec_a: first vector, torch Short Tensor shape (HD_dim,)
		vec_b: second vector, torch Short Tensor shape (HD_dim,)
		Return  
		------
		rel_dist: relative hamming distance 
		'''	
		vec_c = self.xor(vec_a,vec_b)

		rel_dist = torch.sum(vec_c).float() / float(D)

		return rel_dist





def LBP_window_train(EEG,fs, totalNumberBP, label,T,window_dim): 
	''' Function used during training to take a bigger window of EEG and sum all the histograms
	of LBP frequency of 0.5 seconds windows inside.
	Parameters
		----------
	EEG: samples of EEG on multiple channels.
	fs: frequency of the signal.
	T: LBP length + 1
	label: label of the segment--> during test we can pass a random value, it will not be used
	totalNumberBP: dimensionality of the histogram
	Return  
	------
	quueryVector: we save all the histograms here creating a Vector for train of dimension 
	numberOfWindowsx(2^(T-1)xnChannels).
	'''		
	N_channels,learningEnd = EEG.size()

	index = np.arange(0,learningEnd-window_dim,window_dim)
	queeryVector = torch.cuda.FloatTensor(index.size,totalNumberBP*N_channels+1).zero_()
	i = 0
	for iStep in index:
		temp = LBP_extractor(EEG[:,iStep:iStep+window_dim],totalNumberBP,label,T)
		queeryVector[i,:]= temp.t()
		i = i+1
	return queeryVector

def LBP_extractor(EEG,totalNumberBP,label,T): 
	''' starting from EEG data, it creates the histogram of LBP for the target segment.
	Parameters
	----------
	EEG: samples of EEG on multiple channels.
	T: LBP length + 1
	label: label of the segment--> during test we can pass a random value, it will not be used
	totalNumberBP: dimensionality of the histogram
	Return  
	------
	Features_vector: in position 0 we have the label (for the training). In the other position
	the full histogram encoded, i.e. the features.
	'''	
	N_channels,learningEnd = EEG.size()
	Features_vector = torch.cuda.FloatTensor(totalNumberBP*N_channels+1,1).zero_()	
	LBP_weights = torch.cuda.FloatTensor([2**0, 2**1, 2**2, 2**3, 2**4, 2**5])
	for iStep in range(learningEnd-6):
		x = EEG[:,iStep:(iStep+T)].float()
		bp = (torch.add(-x[:,0:T-1], 1,x[:,1:T])>0).float()
		value = torch.sum(torch.mul(LBP_weights,bp), dim=1)+1
		index= torch.add(torch.mul(torch.cuda.FloatTensor(np.array(range(N_channels))),totalNumberBP),value)
		Features_vector[index.long()] = Features_vector[index.long()]+1
	Features_vector[0] = label
	return Features_vector

def LGP_window_train(EEG,fs, totalNumberBP, label,T, window_dim): 
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
	N_channels,learningEnd = EEG.size()

	index = np.arange(0,learningEnd-window_dim,window_dim/2)
	queeryVector = torch.cuda.FloatTensor(index.size,totalNumberBP*N_channels+1).zero_()
	i = 0
	for iStep in index:
		temp = LGP_extractor(EEG[:,iStep:iStep+window_dim],totalNumberBP,label,T)
		queeryVector[i,:]= temp.t()
		i = i+1
	return queeryVector

def LGP_extractor(EEG,totalNumberGP,label,T): 
	''' starting from EEG data, it creates the histogram of LGP for the target segment.
	Parameters
	----------
	EEG: samples of EEG on multiple channels.
	T: LBP length + 1
	label: label of the segment--> during test we can pass a random value, it will not be used
	totalNumberGP: dimensionality of the histogram
	Return  
	------
	Features_vector: in position 0 we have the label (for the training). In the other position
	the full histogram encoded, i.e. the features.
	'''		
	N_channels,learningEnd = EEG.size()
	Features_vector = torch.cuda.FloatTensor(totalNumberGP*N_channels+1,1).zero_()	
	LBP_weights = torch.cuda.FloatTensor([2**0, 2**1, 2**2, 2**3, 2**4, 2**5])
	for iStep in range(4,learningEnd-4):
		x = EEG[:,iStep-(T-1)/2:iStep+(T-1)/2+1].float()
		x = torch.abs(torch.add(x,torch.mul(torch.ones(T,1).cuda(),-x[:,(T-1)/2]).t()))
		bp = (torch.add(x,torch.mul(torch.ones(T,1).cuda(),-torch.mean( x[:,(0, 1, 2, 4,5,6)],1)).t())>0).float()
		value = torch.sum(torch.mul(LBP_weights,bp[:,(0,1,2,4,5,6)]), dim=1)+1
		index= torch.add(torch.mul(torch.cuda.FloatTensor(np.array(range(N_channels))),totalNumberGP),value)
		Features_vector[index.long()] = Features_vector[index.long()]+1
	Features_vector[0] = label
	return Features_vector
