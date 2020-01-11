from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv("./datasets/sonar.csv", header=None)
ds = df.values                                          #데이터프레임에서 값만 불러오기
X = ds[:,0:60]                                          #입력값
Y_obj = ds[:,60] 

e = LabelEncoder()                                      #One_hot_encoding하기
e.fit(Y_obj)
Y = e.transform(Y_obj)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

X_train = X_train.astype('float')
X_test = X_test.astype('float')

model = models.Sequential([layers.Dense(24, input_dim=60, activation='relu'), layers.Dense(10, activation='relu'), layers.Dense(1, activation='sigmoid')])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=130, batch_size=5)

print(model.evaluate(X_test,Y_test))