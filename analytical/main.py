from NN import *  
import matplotlib.pyplot as plt

NN = neuralNet(2)
NN.addLayer(5)
NN.addLayer(5)
NN.addLayer(3)

train_data = []

for i in range(10000):                                           
    tempfeature = np.array([5,5]) + np.random.rand(2) * 10       
    tempfeature = tempfeature.reshape(-1, 1)
    templabel = np.array([1,0,0]).reshape(-1,1)                  
    train_data.append((tempfeature,templabel))
    tempfeature =  np.array([30,30]) + np.random.rand(2) * 10   
    tempfeature = tempfeature.reshape(-1, 1)
    templabel = np.array([0,0,1]).reshape(-1,1)                  
    train_data.append((tempfeature,templabel))
train_data = np.array(train_data)
np.random.shuffle(train_data)                               
forPlotXValue = []
forPlotYValue = []

for i, (feature, label) in enumerate(train_data):
    forPlotXValue.append(feature[0])                            
    forPlotYValue.append(feature[1])                            
    NN.fit(feature,label, 0.01)                               


feature =  np.array([30,30]) + np.random.rand(2) * 10     
feature = tempfeature.reshape(-1, 1)
print("testen vorhersage fur den Punkt [30,30]", NN.propagate(feature))  

plt.scatter(forPlotXValue, forPlotYValue)    
