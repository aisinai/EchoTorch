#!/usr/bin/env python
import sys
import numpy as np
import os
import scipy.io as scp
import time
import pdb
import json

import keras
from keras.utils import np_utils
from keras.models import model_from_json
keras.backend.set_image_data_format('channels_first')
print ('Using Keras image_data_format=%s' % keras.backend.image_data_format())

from load_signals import PrepData
from prep_data import train_val_loo_split, train_val_test_split
from cnn import ConvNN
import h5py
from scipy.signal import stft
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
        EEGSez1 = np.array(f['EEGRefMedian'][:,seizure_begin[ict][0]+seizure_1[0]*fs:seizure_begin[ict][0]+seizure_1[1]*fs])
        EEGSez2 = np.array(f['EEGRefMedian'][:,seizure_begin[ict2][0]+seizure_2[0]*fs:seizure_begin[ict2][0]+seizure_2[1]*fs])
        EEGInterictal = np.array(f['EEGRefMedian'][:,seizure_begin[ict][0]- 10*minutes:seizure_begin[ict][0] - 10*minutes +30*fs])
        #EEGInterictal = torch.from_numpy(np.array(f['EEGRefMedian'][:,seizure_begin[0][0]+120*fs:seizure_begin[0][0]+150*fs])).cuda()
        #EEGInterictal = torch.cat((EEGInterictal, EEGInterictal1), dim = 1)
        #EEGSez1 = EEGSez1[:,np.arange(0,EEGSez1.shape[1],2)]
        #EEGInterictal = EEGInterictal[:,np.arange(0,EEGInterictal.shape[1],2)]
        print('prova')
        T = 7
        totalNumberBP = 2**(T-1)
        D = 10000;              # dimension of hypervector
        nSignals = EEGSez1.shape[0]
        #####TRAINING STEP#######
        t=time.time()
        print('Learning Seizure 1')
        Xt1, yt1 = PrepData(EEG = EEGSez1,frequency = fs, segment = 'ictal').apply()
        Xt2, yt2 = PrepData(EEG = EEGSez2,frequency = fs,segment = 'ictal').apply()
        print('Learning Interictal period')
        Xt3, yt3 = PrepData(EEG = EEGInterictal,frequency = fs,segment = 'interictal').apply()
        #Matrix_train = np.concatenate((temp1, temp2),0)
        #X_train = np.concatenate((Xt1, Xt3),0)
        #y_train = np.concatenate((yt1, yt3),0)
        X_train = np.concatenate((Xt1, Xt2),0)
        y_train = np.concatenate((yt1, yt2),0)
        X_train = np.concatenate((X_train, Xt3),0)
        y_train = np.concatenate((y_train, yt3),0)
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        build_type='test'
        model = ConvNN(patient,batch_size=32,nb_classes=2,epochs=50,mode=build_type)

        model.setup(X_train.shape)
        model.fit(X_train, y_train)
        i = 0
        j = 0
        t = time.time()
        window_dim = fs*9
        #ictal period test
        seizure_slices_seiz = seizure_begin-1*minutes

        prediction = np.zeros(seizure_begin.shape[0]*3*minutes/fs)
        ict_scores = np.zeros(seizure_begin.shape[0]*3*minutes/fs)
        interictal_scores = np.zeros(seizure_begin.shape[0]*3*minutes/fs)
        for seizures in seizure_slices_seiz[:,:]:
            EEGtest=np.array(f['EEGRefMedian'][:,seizures[0]:(seizures[0]+3*minutes)])
            index = np.arange(0,EEGtest.shape[1]-window_dim,fs)
            j = j+1
            for iStep in index:
                fr,t,stft_data = stft(EEGtest[:,iStep:iStep+window_dim],fs)
                stft_data = stft_data[:,0:75,:]
                stft_data = np.transpose(stft_data,(0,2,1))
                stft_data = np.abs(stft_data)+1e-6
                stft_data = np.log10(stft_data)
                indices = np.where(stft_data <= 0)
                stft_data[indices] = 0
                stft_data = stft_data.reshape(-1, 1, stft_data.shape[0],
                                                        stft_data.shape[1],
                                                        stft_data.shape[2])
                X_test = stft_data
                scores = model.predict_proba(X_test)
		ict_scores[i] = scores[0][1]
		interictal_scores[i] = scores[0][0]
                if scores[0][0] > scores[0][1]:
                    prediction[i] = 0
                else:
                    prediction[i] = 1
                i = i + 1
                print ('Pat'+patient+'Seiz: ' + str(j) +'; half second: ' + str(i%171))
        np.save('/usr/scratch/sassauna3/msc18f3/Predictions/Pat'+ patient+'/' + 'CNNSeizures_rel'+'.txt',prediction)

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
        prediction = np.zeros(20*60*minutes/fs)
        ict_scores = np.zeros(20*60*minutes/fs)
        interictal_scores = np.zeros(20*60*minutes/fs)
        for seizures in seizure_slices10:
            EEGtest=np.array(f['EEGRefMedian'][:,seizures:seizures+60*minutes])
            index = np.arange(0,EEGtest.shape[1]-window_dim,fs)
            j = j+1
            for iStep in index:
                fr,t,stft_data = stft(EEGtest[:,iStep:iStep+window_dim],512)
                stft_data = stft_data[:,0:75,:]
                stft_data = np.transpose(stft_data,(0,2,1))
                stft_data = np.abs(stft_data)+1e-6
                stft_data = np.log10(stft_data)
                indices = np.where(stft_data <= 0)
                stft_data[indices] = 0
                stft_data = stft_data.reshape(-1, 1, stft_data.shape[0],
                                                        stft_data.shape[1],
                                                        stft_data.shape[2])
                X_test = stft_data
                scores = model.predict_proba(X_test)
		ict_scores[i] = scores[0][1]
		interictal_scores[i] = scores[0][0]
                if scores[0][0] > scores[0][1]:
                    prediction[i] = 0
                else:
                    prediction[i] = 1
                i = i + 1
                print ('Pat'+patient+'hour: ' + str(j) +'; half second: ' + str(i%3591))
        np.save('/usr/scratch/sassauna3/msc18f3/Predictions/Pat'+ patient+'/' + 'CNN20hours_rel'+'.txt',prediction)

        print(time.time()-t)
