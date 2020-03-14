from NN import *  
import matplotlib.pyplot as plt

NN = neuralNet(2)
NN.addLayer(5)
NN.addLayer(5)
NN.addLayer(3)

train_data = []

for i in range(10000):                                           # ich erstelle punkte in 2d koordinatensystem einmal um den Punkt 5,5 und einmal um den Punkt 30,30
    tempfeature = np.array([5,5]) + np.random.rand(2) * 10       # ich baue hier einen fehler ein fur punkt [5,5]
    tempfeature = tempfeature.reshape(-1, 1)
    templabel = np.array([1,0,0]).reshape(-1,1)                  # bei punkt 5,5 soll das neuronale netz 1,0,0 ausgeben 
    train_data.append((tempfeature,templabel))
    tempfeature =  np.array([30,30]) + np.random.rand(2) * 10   # ich baue hier einen fehler ein fur punkt [30,30]
    tempfeature = tempfeature.reshape(-1, 1)
    templabel = np.array([0,0,1]).reshape(-1,1)                  # bei punkt 30,30 soll das neuronale netz 0,0,1 ausgeben 
    train_data.append((tempfeature,templabel))
train_data = np.array(train_data)
np.random.shuffle(train_data)                               # ich mische den datensatz durch

forPlotXValue = []
forPlotYValue = []

for i, (feature, label) in enumerate(train_data):
    forPlotXValue.append(feature[0])                            # das ist nur fur das ploten fur die darstellung die x punkte im koordinaten system
    forPlotYValue.append(feature[1])                            # das ist nur fur die darstellugn           die y punkte im koordinatensystem
    NN.fit(feature,label, 0.01)                                   # das trainieren des Neuronalen netzes

# du siehst wie der fehler immer weniger wird

feature =  np.array([30,30]) + np.random.rand(2) * 10             # ein testpunkt erstell
feature = tempfeature.reshape(-1, 1)
print("testen vorhersage fur den Punkt [30,30]", NN.propagate(feature))    # sollte [0,0,1] rauskommen ---Wichtig gerundet muss 0,0,1 rauskommen ----

plt.scatter(forPlotXValue, forPlotYValue)    # zum ploten der punkte
plt.show()
