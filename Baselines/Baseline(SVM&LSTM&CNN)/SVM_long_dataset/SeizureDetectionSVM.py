#!/usr/bin/env python
import sys
import numpy as np
import os
import scipy.io as scp
import iEEGFunctions  #here there are all the functions that we use
import torch
import time
import pdb
import json
import matplotlib.pyplot as plt
from sklearn import svm
import h5py

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
        device = 2
        torch.cuda.set_device(device)
        ict = 0
        ict2 = sez_2_choice[patient]
        seizure_1 = [seizure_1_beg[patient], seizure_1_end[patient]]
        seizure_2 = [beg[patient], end[patient]]
        file = patient + '.mat'
        #working_dir = '/usr/scratch/sassauna3/msc18f3/longTermiEEGanonym/'
        working_dir = '/mnt/c/Users/Faried/Desktop/aisinai/EchoTorch/examples/datasets/eegs/longterm'
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
        window_dim = np.int(np.ceil(fs*10/10))
        #seizure_5 = [20,45]
        EEGSez1 = torch.from_numpy(np.array(f['EEGRefMedian'][:,seizure_begin[ict][0]+seizure_1[0]*fs:seizure_begin[ict][0]+seizure_1[1]*fs])).cuda()
        EEGSez2 = torch.from_numpy(np.array(f['EEGRefMedian'][:,seizure_begin[ict2][0]+seizure_2[0]*fs:seizure_begin[ict2][0]+seizure_2[1]*fs])).cuda()
        EEGInterictal = torch.from_numpy(np.array(f['EEGRefMedian'][:,seizure_begin[ict][0]- 10*minutes:seizure_begin[ict][0] - 10*minutes +30*fs])).cuda()
        #EEGInterictal = torch.from_numpy(np.array(f['EEGRefMedian'][:,seizure_begin[0][0]+120*fs:seizure_begin[0][0]+150*fs])).cuda()
        #EEGInterictal = torch.cat((EEGInterictal, EEGInterictal1), dim = 1)
        print('prova')
        T = 7
        totalNumberBP = 2**(T-1)
        D = 10000;              # dimension of hypervector
        nSignals = EEGSez1.shape[0]
        torch.manual_seed(1)
        #####TRAINING STEP#######
        t=time.time()
        print('Learning Seizure 1')
        temp1=iEEGFunctions.LBP_window_train(EEGSez1,fs, totalNumberBP, 1,T,window_dim)
        temp2=iEEGFunctions.LBP_window_train(EEGSez2,fs, totalNumberBP, 1,T,window_dim)
        print('Learning Interictal period')
        temp3=iEEGFunctions.LBP_window_train(EEGInterictal,fs, totalNumberBP, 0,T,window_dim)
        #queeryVectorS0 = torch.sign(torch.add(torch.add(temp1,1,temp2),1,torch.mul(temp1,temp2)))
        Matrix_train = np.concatenate((temp1, temp2),0)
        #Matrix_train = np.concatenate((temp1, temp3),0)
        Matrix_train = np.concatenate((Matrix_train, temp3),0)
        Matrix_train = Matrix_train[np.random.permutation(len(Matrix_train))]
        y_train = Matrix_train[:,0]
        X_train = Matrix_train[:,1:totalNumberBP*nSignals+1]
        clf = svm.SVC(kernel='linear', C = 1.0) #if results are different make c = 0.1
        clf.fit(X_train,y_train)
        ###TESTING####
        i = 0
        j = 0
        t = time.time()
        #ictal period test: SVM SEIZURES are saved under MLPSeizures
        seizure_slices_seiz = seizure_begin-1*minutes
        prediction = np.zeros(seizure_begin.shape[0]*3*minutes/fs*2)
        scores = np.zeros(seizure_begin.shape[0]*3*minutes/fs*2)
        for seizures in seizure_slices_seiz:
            EEGtest=torch.from_numpy(np.array(f['EEGRefMedian'][:,seizures[0]:seizures[0]+3*minutes])).cuda()
            index = np.arange(0,EEGtest.size(1)-fs,fs/2)
            j = j+1
            for iStep in index:
                temp = iEEGFunctions.LBP_extractor(EEGtest[:,iStep:iStep+window_dim],totalNumberBP,2,T)
                X_test = temp[1:totalNumberBP*nSignals+1]
                scores[i] = clf.decision_function(X_test.t())[0]
                prediction[i] = clf.predict(X_test.t())
                i = i + 1
                print 'Seiz: ' + str(j) +'; half second: ' + str(i%360)
        np.save('/mnt/c/Users/Faried/Desktop/aisinai/EchoTorch/examples/datasets/eegs/preds/Pat'+ patient+'/' + 'SVMSeizures_rel'+'.txt',prediction)

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
        scores = np.zeros(20*60*minutes/fs*2)
        for seizures in seizure_slices10:
            EEGtest=torch.from_numpy(np.array(f['EEGRefMedian'][:,seizures:seizures+60*minutes])).cuda()
            index = np.arange(0,EEGtest.size(1)-fs,fs/2)
            j = j+1
            for iStep in index:
                temp = iEEGFunctions.LBP_extractor(EEGtest[:,iStep:iStep+window_dim],totalNumberBP,2,T)
                X_test = temp[1:totalNumberBP*nSignals+1]
                scores[i] = clf.decision_function(X_test.t())[0]
                prediction[i] = clf.predict(X_test.t())
                i = i + 1
                print 'Pat'+patient+'hour: ' + str(j) +'; half second: ' + str(i%7200)
        np.save('/mnt/c/Users/Faried/Desktop/aisinai/EchoTorch/examples/datasets/eegs/preds/Pat'+ patient+'/' + 'SVM20hours_rel'+'.txt',prediction)
