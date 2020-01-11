from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)                                #일정결과값으 얻기 위한 시드값 설정

df = pd.read_csv("./datasets/sonar.csv", header=None)   #데이터프레임 불러오기

ds = df.values                                          #데이터프레임에서 값만 불러오기
X = ds[:,0:60]                                          #입력값
Y_obj = ds[:,60]                                        #출력값

e = LabelEncoder()                                      #One_hot_encoding하기
e.fit(Y_obj)
Y = e.transform(Y_obj)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

# model = Sequential()                                    #모델설정하기
# model.add(Dense(24, input_dim=60, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# == 아래와 같이 실행해도 같음
model = Sequential([Dense(24, input_dim=60, activation='relu'), Dense(10, activation='relu'), Dense(1, activation='sigmoid')])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=130, batch_size=5)

print(model.evaluate(X_test,Y_test))