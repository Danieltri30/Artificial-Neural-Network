import tensorflow as tf
import pandas_datareader as pr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import math
import pandas as pd 
import keras_tuner as kt
import numpy as np 
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
print("Keras imports are working!")



"""data = pd.read_csv("apartments_for_rent_classified_100K.csv" , encoding ="utf-8")"""
x_train = pd.read_csv('X_train.csv')
x_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')
y_train = pd.read_csv('y_train.csv')
step = 365

def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, input_shape= (step,1)))
        self.model.add(LSTM(64, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.compile(
            optimizer='adam' , loss = 'mean_squared_error',
            metrics=['mae'])
        self.model.summary()

class NeuralNetwork:

    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        self.model = Sequential()

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, input_shape= (step,1)))
        self.model.add(LSTM(64, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.compile(
            optimizer='adam' , loss = 'mean_squared_error',
            metrics=['mae'])
        self.model.summary()
  

    def train(self, epochs = 50 , batch_size = 32):
        self.model.fit(self.X_train , self.y_train, validation_data = (self.X_test , self.y_test) , epochs = epochs , batch_size = batch_size)

    def evaluate(self): 
        result = self.model.evaluate(self.X_test , self.y_test)
        if len(result) == 2:
            loss,mae = result
            print(f"Loss : {loss} , MAE : {mae}")
        else:
            loss = result
            print(f"Loss: {loss}")



def main():

    tuner = kt.Hyperband(
        build_model,
        objective='val_mae',
        max_epochs=50,
        directory='tuner_results',
        project_name='apartment_model'
    )

   
    tuner.search(x_train, y_train, validation_data=(x_test, y_test), batch_size=32)

   
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    nn = NeuralNetwork(x_train, x_test, y_train, y_test)
    nn.build_model()  
    nn.train()
    nn.evaluate()



if __name__ == "__main__":
    main()
                

