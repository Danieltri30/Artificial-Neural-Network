import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
import pandas as pd 
import keras_tuner as kt
import numpy as np 
print(tf.__version__)
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
print("Keras imports are working!")



"""data = pd.read_csv("apartments_for_rent_classified_100K.csv" , encoding ="utf-8")"""
x_train = pd.read_csv('X_train.csv')
x_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')
y_train = pd.read_csv('y_train.csv')

def build_model(hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units', min_value=32, max_value=128, step=32), activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model  


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

    def b_model(self, hp=None):
        self.model = Sequential()
        units = hp.Int('units', min_value=32, max_value=128, step=32) if hp else 64  # Default to 64 units if hp is None
        self.model.add(Dense(units=units, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
  

    def train(self, epochs = 50 , batch_size = 32):
        self.model.fit(self.X_train , self.y_train, validation_data = (self.X_test , self.y_test) , epochs = epochs , batch_size = batch_size)

    def evaluate(self): 

        loss , mae = self.model.evaluate(self.X_test , self.y_test)
        print(f"MAE : {mae}")



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
    nn.b_model(hp=best_hps)  
    nn.train()
    nn.evaluate()



if __name__ == "__main__":
    main()
                

