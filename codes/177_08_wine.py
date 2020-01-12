from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_pre = pd.read_csv('./datasets/wine.csv', header=None)
df = df_pre.sample(frac=1)
ds = df.values
X = ds[:,0:12]
Y = ds[:,12]

model = Sequential()                                    #모델설정하기
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=200, batch_size=200)

print("\n Accuracy: %.4f" %(model.evaluate(X,Y, verbose=0)[1]))