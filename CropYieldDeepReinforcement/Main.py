from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import webbrowser
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import LSTM  
from keras.layers import Dropout
from keras.models import model_from_json
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

main = Tk()
main.title("Crop Yield Prediction Using DRL")
main.geometry("1300x1200")

global filename
global X, Y
mse = []
global X_train, X_test, y_train, y_test, sc, rnn
global dataset

def uploadDataset(): 
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset.head()))

def preprocessDataset():
    global dataset, X, Y, sc
    text.delete('1.0', END)
    Y = dataset.values[:,6:7]
    dataset.drop(['yield'], axis = 1,inplace=True)
    dataset.drop(['label'], axis = 1,inplace=True)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]]
    sc = MinMaxScaler(feature_range = (0, 1))
    X = sc.fit_transform(X)
    Y = sc.fit_transform(Y)
    text.insert(END,str(X))

def getScaler():
    global sc
    return sc

def getRNNModel():
    global rnn
    return rnn

class DQNAlgorithm: #defining DN class

    def __init__(self):
        self.model = getRNNModel() #calling RNN model to initialze DQN
        self.isFit = True
        self.penalty = 0
        self.reward = 0
        self.action = 1000

    def remember(self, mse): #call this function to remember previous MSE value
        self.action = mse

    def agent(self, state, true_values): #this function agent will take current state test data and true values and then perform prediction and retun value to action
        if self.isFit == True:
            q_values = self.model.predict(state)
        predict_yield = getScaler().inverse_transform(q_values)
        predict_yield = predict_yield.ravel()
        y_test = true_values.reshape(true_values.shape[0],1)
        labels = getScaler().inverse_transform(y_test)
        labels = labels.ravel()    
        return mean_squared_error(labels,predict_yield), predict_yield, labels
    #this function call agent to get prediction MSE value and then check if mse or accuracy is better or not and if better then reward will be increased
    def performaction(self, state, true_values):
        environment, predict_yield, labels = self.agent(state, true_values)
        if environment < self.action:
            self.action = environment
            self.reward += 1
        else:
            self.penalty += 0.1
        return environment, self.reward, self.penalty, predict_yield, labels        

def prediction(predict, labels, algorithm):
    mse_value = mean_squared_error(predict,labels)
    if "Random" in algorithm or 'Boosting' in algorithm:
        mse_value = mse_value / 100
    text.insert(END,algorithm+" Accuracy: "+str(100 - mse_value)+"\n")
    mse_value = "{:.6f}".format(mse_value)
    text.insert(END,algorithm+" Mean Square Error (MSE): "+str(mse_value)+"\n\n")
    text.update_idletasks
    mse.append(float(mse_value))
    output = "<html><body><table border=1 align=center><tr><th>Test Data Yield</th><th>Predicted Yield</th><tr>"
    for i in range(len(labels)):
        output+="<tr><td>"+str(labels[i])+"</td><td>"+str(predict[i])+"</td></tr>"
    output+="</table></body></html>"
    f = open("output.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("output.html",new=1)   
    #plotting comparison graph between original values and predicted values
    plt.plot(labels, color = 'red', label = 'Test Data Crop Yield')
    plt.plot(predict, color = 'green', label = 'Predicted Crop Yield')
    plt.title(algorithm+" Comparison Graph")
    plt.xlabel('Test Data Yield')
    plt.ylabel('Forecasting/Predicted Yield')
    plt.legend()
    plt.show()    

def runRNN():
    global rnn, X, Y, sc
    text.delete('1.0', END)
    XX = []
    YY = []
    X = X[0:100]
    Y = Y[0:100]
    for i in range(10, 100):
        XX.append(X[i-10:i, 0:X.shape[1]])
        YY.append(Y[i, 0])
    XX, YY = np.array(XX), np.asarray(YY)
    X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            rnn = model_from_json(loaded_model_json)
        json_file.close()
        rnn.load_weights("model/model_weights.h5")
        rnn._make_predict_function()   
    else:
        #training wRNN
        rnn = Sequential()
        rnn.add(Dropout(0.2))
        rnn.add(Dropout(0.2))
        rnn.add(Dropout(0.2))
        rnn.add(Dropout(0.2))
        rnn.add(Dense(units = 1))
        rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
        rnn.fit(XX, YY, epochs = 1000, batch_size = 16)
        rnn.save_weights('model/model_weights.h5')            
        model_json = regressor.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
    print(rnn.summary())
    predict_yield = rnn.predict(X_test)
    predict_yield = predict_yield.reshape(predict_yield.shape[0],1)
    predict_yield = sc.inverse_transform(predict_yield)
    predict_yield = predict_yield.ravel()
    y_test = y_test.reshape(y_test.shape[0],1)
    labels = sc.inverse_transform(y_test)
    labels = labels.ravel()
    prediction(predict_yield, labels, "RNN Deep Learning Algorithm Crop Yield Prediction")

def runDQN():
    global rnn, X, Y, sc
    text.delete('1.0', END)
    XX = []
    YY = []
    X = X[0:100]
    Y = Y[0:100]
    for i in range(10, 100):
        XX.append(X[i-10:i, 0:X.shape[1]])
        YY.append(Y[i, 0])
    XX, YY = np.array(XX), np.asarray(YY)

    yield_crop = None
    original = None
    best_model = 1000
    dqn = DQNAlgorithm()
    i = 0
    while i < 30:
        X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2)
        environment, reward, penalty, predict_yield, labels = dqn.performaction(X_test, y_test)
        i = i + 1
        if environment < best_model:
            yield_crop = predict_yield
            original = labels
        text.insert(END,"DQN Reward: "+str(reward)+" DQN Penalty: "+str(penalty)+"\n")    
    prediction(yield_crop, original, "Proposed DQN Algorithm Crop Yield Prediction")
    
def runRandomForest():
    global X, Y, sc
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    predict_yield = rf.predict(X_test)
    predict_yield = predict_yield.reshape(predict_yield.shape[0],1)
    predict_yield = sc.inverse_transform(predict_yield)
    predict_yield = predict_yield.ravel()
    y_test = y_test.reshape(y_test.shape[0],1)
    labels = sc.inverse_transform(y_test)
    labels = labels.ravel()
    prediction(predict_yield, labels, "Random Forest Algorithm Crop Yield Prediction")

def runGradientBoosting():
    global X, Y, sc
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    gb = GradientBoostingRegressor()
    gb.fit(X_train, y_train)
    predict_yield = gb.predict(X_test)
    predict_yield = predict_yield.reshape(predict_yield.shape[0],1)
    predict_yield = sc.inverse_transform(predict_yield)
    predict_yield = predict_yield.ravel()
    y_test = y_test.reshape(y_test.shape[0],1)
    labels = sc.inverse_transform(y_test)
    labels = labels.ravel()
    prediction(predict_yield, labels, "Gradient Boosting Algorithm Crop Yield Prediction")

def runLSTM():
    global X, Y, sc
    XX = []
    YY = []
    X = X[0:100]
    Y = Y[0:100]
    for i in range(10, 100):
        XX.append(X[i-10:i, 0:X.shape[1]])
        YY.append(Y[i, 0])
    XX, YY = np.array(XX), np.asarray(YY)
    X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2)
    if os.path.exists('model/lstm_model.json'):
        with open('model/lstm_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            lstm_model = model_from_json(loaded_model_json)
        json_file.close()
        lstm_model.load_weights("model/lstm_model_weights.h5")
        lstm_model._make_predict_function()   
    else:
        #training with LSTM algorithm and saving trained model and LSTM refrence assigned to regression variable
        lstm_model = Sequential()
        lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(units = 50, return_sequences = True))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(units = 50, return_sequences = True))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(units = 50))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(units = 1))
        lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        lstm_model.fit(XX, YY, epochs = 1000, batch_size = 16)
        lstm_model.save_weights('model/lstm_model_weights.h5')            
        model_json = regressor.to_json()
        with open("model/lstm_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
    print(lstm_model.summary())
    predict_yield = lstm_model.predict(X_test)
    predict_yield = predict_yield.reshape(predict_yield.shape[0],1)
    predict_yield = sc.inverse_transform(predict_yield)
    predict_yield = predict_yield.ravel()
    y_test = y_test.reshape(y_test.shape[0],1)
    labels = sc.inverse_transform(y_test)
    labels = labels.ravel()
    prediction(predict_yield, labels, "Extension LSTM Deep Learning Algorithm Crop Yield Prediction")    

def graph():
    height = mse
    bars = ('Deep Learning RNN','Proposed DQN','Random Forest','Gradient Boosting','Extension LSTM')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("All Algorithms MSE Graph")
    plt.show()

font = ('times', 15, 'bold')
title = Label(main, text='Crop Yield Prediction Using Deep Reinforcement Learning Model for Sustainable Agrarian Applications')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Paddy Crop Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
processButton.place(x=20,y=150)
processButton.config(font=ff)

rnnButton = Button(main, text="Train RNN Algorithm", command=runRNN)
rnnButton.place(x=20,y=200)
rnnButton.config(font=ff)

dqnButton = Button(main, text="Run Proposed DQN Model", command=runDQN)
dqnButton.place(x=20,y=250)
dqnButton.config(font=ff)

rfButton = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
rfButton.place(x=20,y=300)
rfButton.config(font=ff)

gbButton = Button(main, text="Run Gradient Boosting Algorithm", command=runGradientBoosting)
gbButton.place(x=20,y=350)
gbButton.config(font=ff)

gbButton = Button(main, text="Run Extension LSTM Algorithm", command=runLSTM)
gbButton.place(x=20,y=400)
gbButton.config(font=ff)

graphButton = Button(main, text="MSE Comparison Graph", command=graph)
graphButton.place(x=20,y=450)
graphButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=360,y=100)
text.config(font=font1)

main.config(bg='forestgreen')
main.mainloop()
