import torch
import pdb
import matplotlib.pyplot as plt
import numpy as np

torch.cuda.set_device(3)
number_seizures = range(0,110)
prediction = torch.zeros(1,1)
prediction = prediction[0,:]

threshold = -9.5/10
#seizures

prediction=torch.from_numpy(np.load('/usr/scratch/sassauna3/msc18f3/Predictions/Pat16/' + 'LSTMSeizures'+'.txt.npy')).cuda()
Pred_mean = torch.FloatTensor(prediction.size()[0]).zero_()
for i in np.arange(10,prediction.size()[0]):
    Pred_mean[i] = torch.mean(prediction[i-10:i])
    print i
Pred_mean = torch.add(Pred_mean, threshold )

Final_prediction = torch.sign(Pred_mean)
index = np.arange(0,Final_prediction.shape[0], 360)
Delay = np.zeros(index.size)
miss = 0
for i in index:
	aa = np.where(Final_prediction[i+120:i+360]+1)
	if aa[0].size == 0:
		miss +=1
	else:
		Delay[int(i/360)] = (aa[0][0])/2
print 'Delay: ' + str(np.sum(Delay[1:])/(Delay.size-1-miss))+ '\t Seizure missed: ' + str(miss)
pdb.set_trace()
x = np.arange(0.0,(Final_prediction.size()[0])/float(2*60),1/float(2*60))
plt.scatter(x,Final_prediction, c = 'w', edgecolors = 'k')
plt.xlabel('time[minutes]')
plt.ylabel('Prediction')
plt.title('Patient4: Final Prediction ')
plt.show()

#false alarms
prediction=torch.from_numpy(np.load('/usr/scratch/sassauna3/msc18f3/Predictions/Pat19/' + 'LSTM20hours'+'.txt.npy')).cuda()
Pred_mean = torch.FloatTensor(prediction.size()[0]).zero_()
for i in np.arange(10,prediction.size()[0]):
     Pred_mean[i] = torch.mean(prediction[i-10:i])
     print i
Pred_mean = torch.add(Pred_mean, threshold )

Final_prediction = torch.sign(Pred_mean)
x = np.arange(0.0,(Final_prediction.size()[0])/float(2*60),1/float(2*60))
plt.scatter(x,Final_prediction, c = 'w', edgecolors = 'k')
plt.xlabel('time[minutes]')
plt.ylabel('Prediction')
plt.title('Patient4: Final Prediction ')
plt.show()
