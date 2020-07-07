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

prediction=torch.from_numpy(np.load('/usr/scratch/sassauna3/msc18f3/Predictions/Pat22/' + 'CNNSeizures'+'.txt.npy')).cuda()
Pred_mean = torch.FloatTensor(prediction.size()[0]).zero_()
for i in np.arange(10,prediction.size()[0]):
    Pred_mean[i] = torch.mean(prediction[i-10:i])
    print i
Pred_mean = torch.add(Pred_mean, threshold)

Final_prediction = torch.sign(Pred_mean)
index = np.arange(0,Final_prediction.shape[0], 180)
Delay = np.zeros(index.size)
miss = 0
for i in index:
	aa = np.where(Final_prediction[i+51:i+180]+1)
	if aa[0].size == 0:
		miss +=1
	else:
		Delay[int(i/180)] = (aa[0][0])
print 'Delay: ' + str(np.sum(Delay[1:])/(Delay.size-1-miss))+ '\t Seizure missed: ' + str(miss)
pdb.set_trace()
x = np.arange(0.0,(Final_prediction.size()[0])/float(60),1/float(60))
plt.scatter(x,Final_prediction, c = 'w', edgecolors = 'k')
plt.xlabel('time[minutes]')
plt.ylabel('Prediction')
plt.title('Patient4: Final Prediction ')
plt.show()

#false alarms
prediction=torch.from_numpy(np.load('/usr/scratch/sassauna3/msc18f3/Predictions/Pat22/' + 'CNN20hours'+'.txt.npy')).cuda()
Pred_mean = torch.FloatTensor(prediction.size()[0]).zero_()
for i in np.arange(10,prediction.size()[0]):
    Pred_mean[i] = torch.mean(prediction[i-10:i])
    print i
Pred_mean = torch.add(Pred_mean, threshold)

Final_prediction = torch.sign(Pred_mean)
x = np.arange(0.0,(Final_prediction.size()[0])/float(60),1/float(60))
plt.scatter(x,Final_prediction, c = 'w', edgecolors = 'k')
plt.xlabel('time[minutes]')
plt.ylabel('Prediction')
plt.title('Patient4: Final Prediction ')
plt.show()
