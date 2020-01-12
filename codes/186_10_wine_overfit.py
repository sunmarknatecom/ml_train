from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

df_pre = pd.read_csv('./datasets/wine.csv', header=None)
df = df_pre.sample(frac=0.15)

ds = df.values
X = ds[:,0:12]
Y = ds[:,12]

model = Sequential()                                    #모델설정하기
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

MODEL_DIR = './model_02/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "./model_02/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)


history = model.fit(X, Y, validation_split=0.33, epochs=3500, batch_size=500)

y_vloss = history.history['val_loss']

y_acc = history.history['accuracy']

x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vloss, "o", c='red', markersize=3)
plt.plot(x_len, y_acc, "o", c='blue', markersize=3)

plt.show()

print("\n Accuracy: %.4f" %(model.evaluate(X,Y, verbose=0)[1]))