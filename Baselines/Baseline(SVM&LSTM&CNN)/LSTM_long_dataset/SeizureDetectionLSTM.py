#!/usr/bin/env python
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import sys
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import h5py
import FunctionsNN  #here there are all the functions that we use
import time
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten, TimeDistributed, GlobalMaxPooling1D
from keras.utils import np_utils

if __name__ == '__main__':
    fs_choice = {'05':512, '11':1024}
    #offset for second training seizure: patient 12 and 17 use the second seizure only for validation, but no segment from them is used for training
    beg = {'04':12, '07':10,'12':28, '08':30}
    end = {'04':43, '07':40,'12':45, '08':60}
    #second training seizure: is always the second seizure. Only patient 04 makes exception, because first 3 seizures were almost identical
    seiz_2_choice = {'04':3, '07':1,'12':1, '08': 67}
    train_seiz = {'01':1,'02':1,'03':1,'04':2,'05':1,'06':1,'07':2,'08':2,'09':2,'10':1,'11':1,'12':2,'13':2,'14':1,'15':1,'16':1,'17':1,'18':1}
    patients_list = train_seiz.keys()
    #offset from begin of first seizure to define the training segment: we train using segments from seizure_1_beg to seizure_1_end
    seizure_1_beg = {'01':10, '02':10, '03':10, '04':35, '05':10, '06':3, \
    '07':4,'08':0, '09':0, '10':20, '11':0, '12':33, '13':12, '14':15, '15':5, '16':10, '17':13, '18':50}
    seizure_1_end = {'01':40, '02':30, '03':30, '04':43, '05':16, '06':30,\
    '07':12, '08':6, '09':8, '10':54, '11':30, '12':58, '13':35, '14':30, '15':40, '16':40, '17':40, '18':80}
    for patient in patient_list:
        ict = 0
        ict2 = sez_2_choice[patient]
        seizure_1 = [seizure_1_beg[patient], seizure_1_end[patient]]
        seizure_2 = [beg[patient], end[patient]]
        file = patient + '.mat'
        #working_dir = '/usr/scratch/sassauna3/msc18f3/longTermiEEGanonym/'
        working_dir = '/usr/scratch/sassauna3/msc18f3/longTermiEEGanonym/'
        loading_file = working_dir + file
        f = h5py.File(loading_file, 'r')
        if patient == '07' or patient == '14':
            fs = fs_choice[patient]
        else:
            fs = np.array(f['fs']).astype(int)
            fs = fs[0][0]
        #fs = fs_choice[patient]
        seizure_begin = (np.array(f['timeCollStart'])*fs).astype(int)
        #offset from begin of the seizure to train
        second = fs
        minutes = second*60
        #seizure_5 = [20,45]
        EEGSez1 = np.array(f['EEGRefMedian'][:,seizure_begin[ict][0]+seizure_1[0]*fs:seizure_begin[ict][0]+seizure_1[1]*fs])
        EEGSez2 = np.array(f['EEGRefMedian'][:,seizure_begin[ict2][0]+seizure_2[0]*fs:seizure_begin[ict2][0]+seizure_2[1]*fs])
        EEGInterictal = np.array(f['EEGRefMedian'][:,seizure_begin[ict][0]- 10*minutes:seizure_begin[ict][0] - 10*minutes +30*fs])
        #EEGInterictal = torch.from_numpy(np.array(f['EEGRefMedian'][:,seizure_begin[0][0]+120*fs:seizure_begin[0][0]+150*fs])).cuda()
        #EEGInterictal = torch.cat((EEGInterictal, EEGInterictal1), dim = 1)
	#if fs==1024:
	#	EEGSez1 = EEGSez1[:,np.arange(0,EEGSez1.shape[1],2)]
	#	#EEGSez2 = EEGSez2[:,np.arange(0,EEGSez2.shape[1],2)]
	#	fs = 512
        #	EEGInterictal = EEGInterictal[:,np.arange(0,EEGInterictal.shape[1],2)]
        #fs = 512
        print('prova')
        T = 7
        totalNumberBP = 2**(T-1)
        D = 10000;              # dimension of hypervector
        N_channels = EEGSez1.shape[0]
        #####TRAINING STEP#######
        t=time.time()
        print('Learning Seizure 1')
        temp1=FunctionsNN.window_train_creation(EEGSez1,fs, 1)
        temp2=FunctionsNN.window_train_creation(EEGSez2,fs, 1)
        print('Learning Interictal period')
        temp3=FunctionsNN.window_train_creation(EEGInterictal,fs, 0)
        #queeryVectorS0 = torch.sign(torch.add(torch.add(temp1,1,temp2),1,torch.mul(temp1,temp2)))
        #Matrix_train = np.concatenate((temp1, temp3),0)
        Matrix_train = np.concatenate((temp1, temp2),0)
        Matrix_train = np.concatenate((Matrix_train, temp3),0)
        Matrix_train = Matrix_train[np.random.permutation(len(Matrix_train))]
        y_train = Matrix_train[:,fs*N_channels]
        X_train = Matrix_train[:,0:fs*N_channels]
        data_length = fs*N_channels
        timesteps = fs/2*N_channels
        data_dim = data_length//timesteps
        X_train=X_train.reshape([X_train.shape[0], timesteps, data_dim])
        y_train=np_utils.to_categorical(y_train, num_classes=2)
        model = Sequential()
        model.add(LSTM(100, input_shape= (timesteps, data_dim), return_sequences = True))
        model.add(TimeDistributed(Dense(50)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(X_train, y_train,  batch_size=64, epochs=80)
        model_json = model.to_json()
        with open("LSTM.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("/usr/scratch/sassauna3/msc18f3/LSTM_24.h5")
        print("Saved model to disk")
        ####TESTING####
        i = 0
        j = 0
        #ictal period test
        seizure_slices_seiz = seizure_begin-1*minutes
        prediction = np.zeros(seizure_begin.shape[0]*3*minutes/fs*2)
        ict_scores = np.zeros(seizure_begin.shape[0]*3*minutes/fs*2)
        interictal_scores = np.zeros(seizure_begin.shape[0]*3*minutes/fs*2)
        for seizures in seizure_slices_seiz[:,:]:
        	EEGtest=np.array(f['EEGRefMedian'][:,seizures[0]:seizures[0]+3*minutes])
        	index = np.arange(0,EEGtest.shape[1]-fs,fs/2)
        	j = j+1
        	for iStep in index:
        		X_test = EEGtest[:,iStep:iStep+fs].reshape(N_channels*fs)
        		X_test=X_test.reshape([1, timesteps, data_dim])
        		scores = model.predict(X_test)
		    
                ict_scores[i] = scores[0][1]
			    interictal_scores[i] = scores[0][0]
        	
            	if scores[0][0] > scores[0][1]:
        			prediction[i] = 0
        		else:
        			prediction[i] = 1
        		i = i + 1
        		print 'Pat'+patient+'Seiz: ' + str(j) +'; half second: ' + str(i%360)
        np.save('/mnt/c/Users/Faried/Desktop/aisinai/EchoTorch/examples/datasets/eegs/preds'+ patient+'/' + 'LSTMSeizures_rel'+'.txt',prediction)

        #interictal period test
        i = 0
        j = 0
        begin_test = 0
        end_test = begin_test + f['EEGRefMedian'].shape[1]
        seizure_slices_prov = np.arange(begin_test,end_test,60*minutes)
	np.random.seed(0)
        seizure_slices_prov = seizure_slices_prov[np.random.permutation(len(seizure_slices_prov))]
        seizure_slices = []
        invalid = 0
        for slices in seizure_slices_prov:
            for seiz_beg in seizure_begin:
                if seiz_beg > slices and seiz_beg < slices+60*minutes:
                    invalid = 1
            if invalid == 0:
                seizure_slices.append(slices)
            invalid = 0
        seizure_slices10 = seizure_slices[0:20]
        prediction = np.zeros(20*60*minutes/fs*2)
        ict_scores = np.zeros(20*60*minutes/fs*2)
        interictal_scores = np.zeros(20*60*minutes/fs*2)

        for seizures in seizure_slices10:
            EEGtest=np.array(f['EEGRefMedian'][:,seizures:seizures+60*minutes])
            index = np.arange(0,EEGtest.shape[1]-fs,fs/2)
            j = j+1
            for iStep in index:
                X_test = EEGtest[:,iStep:iStep+fs].reshape(N_channels*fs)
                X_test=X_test.reshape([1, timesteps, data_dim])
                scores = model.predict(X_test)
		ict_scores[i] = scores[0][1]
		interictal_scores[i] = scores[0][0]
                if scores[0][0] > scores[0][1]:
                    prediction[i] = 0
                else:
                    prediction[i] = 1
                i = i + 1
                print 'Pat'+patient+'hour: ' + str(j) +'; half second: ' + str(i%7200)
        np.save('/usr/scratch/sassauna3/msc18f3/Predictions/Pat'+ patient+'/' + 'LSTM20hours_rel'+'.txt',prediction)

        print(time.time()-t)
