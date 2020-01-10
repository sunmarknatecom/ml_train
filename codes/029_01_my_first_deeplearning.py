from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

ds = np.loadtxt("./datasets/thoracicsurgery.csv", delimiter=",")

X = ds[:,0:17]
Y = ds[:,17]

model = Sequential([Dense(30, input_dim=17, activation='relu'), Dense(1, activation='sigmoid')])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=10)

print(model.evaluate(X,Y,verbose=0))