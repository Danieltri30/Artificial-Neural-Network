import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
import keras_tuner as kt
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

# Load data
x_train = pd.read_csv('X_train.csv').to_numpy()
x_test = pd.read_csv('X_test.csv').to_numpy()
y_train = pd.read_csv('y_train.csv').to_numpy()
y_test = pd.read_csv('y_test.csv').to_numpy()

# Parameters
params = {"input_w": 330, "input_h": 132, "num_classes": 3, "batch_size": 1024, "epochs": 200}
x_train = np.reshape(x_train, (x_train.shape[0], int(np.sqrt(x_train.shape[1])), int(np.sqrt(x_train.shape[1])), 1))
x_train_resized = tf.image.resize(x_train, [330, 132])
x_train = x_train_resized
x_test = x_test.reshape((-1, params["input_w"], params["input_h"], 1))
y_train = to_categorical(y_train, num_classes=params["num_classes"])
y_test = to_categorical(y_test, num_classes=params["num_classes"])

# Build model for Tuner
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(params["input_w"], params["input_h"], 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Flatten())
    model.add(Dense(hp.Int('units', min_value=64, max_value=256, step=64), activation='relu'))
    model.add(Dropout(hp.Float('dropout_dense', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(params["num_classes"], activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Tuner setup
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    directory='tuner_results',
    project_name='apartment_model'
)

# Tuner search
tuner.search(x_train, y_train, validation_data=(x_test, y_test), batch_size=32)

# Best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

# Train best model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=params["epochs"], batch_size=params["batch_size"])