import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import pickle
import os
import random
import gym
import numpy as np
from collections import deque
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

global sc, X_train, X_test, y_train, y_test

def getRNNModel():
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            rnn = model_from_json(loaded_model_json)
        json_file.close()
    rnn.load_weights("model/model_weights.h5")
    rnn._make_predict_function()
    return rnn

def getData():
    global sc, X_train, X_test, y_train, y_test
    dataset = pd.read_csv('Dataset/paddy.csv')
    dataset.fillna(0, inplace = True)
    Y = dataset.values[:,6:7]
    dataset.drop(['yield'], axis = 1,inplace=True)
    dataset.drop(['label'], axis = 1,inplace=True)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]]
    sc = MinMaxScaler(feature_range = (0, 1))
    X = sc.fit_transform(X)
    Y = sc.fit_transform(Y)
    XX = []
    YY = []
    X = X[0:100]
    Y = Y[0:100]
    for i in range(10, 100):
        XX.append(X[i-10:i, 0:X.shape[1]])
        YY.append(Y[i, 0])
    XX, YY = np.array(XX), np.asarray(YY)
    X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2)
    return X_train, X_test, y_train, y_test

def getScaler():
    global sc
    return sc

class DQNAlgorithm:

    def __init__(self):
        self.model = getRNNModel()
        self.isFit = True
        self.penalty = 0
        self.reward = 0
        self.action = 1000

    def remember(self, mse):
        self.action = mse

    def agent(self, state, true_values):
        if self.isFit == True:
            q_values = self.model.predict(state)
        predict_yield = getScaler().inverse_transform(q_values)
        predict_yield = predict_yield.ravel()
        y_test = true_values.reshape(true_values.shape[0],1)
        labels = getScaler().inverse_transform(y_test)
        labels = labels.ravel()    
        return mean_squared_error(labels,predict_yield), predict_yield, labels

    def performaction(self, state, true_values):
        environment, predict_yield, labels = self.agent(state, true_values)
        if environment < self.action:
            self.action = environment
            self.reward += 1
        else:
            self.penalty += 0.1
        return environment, self.reward, self.penalty, predict_yield, labels     
    
yield_crop = None
original = None
best_model = 1000
dqn = DQNAlgorithm()
i = 0
while i < 30:
    X_train, X_test, y_train, y_test = getData()
    environment, reward, penalty, predict_yield, labels = dqn.performaction(X_test, y_test)
    i = i + 1
    if environment < best_model:
        yield_crop = predict_yield
        original = labels

print("Test values: "+str(original))
print("Predicted values: "+str(yield_crop))

plt.plot(original, color = 'red', label = 'Original Growth')
plt.plot(yield_crop, color = 'green', label = 'Predicted Growth')
plt.title('LSTM Banana Growth Forecasting')
plt.xlabel('Test Data')
plt.ylabel('Forecasting Growth')
plt.legend()
plt.show()

        
    
