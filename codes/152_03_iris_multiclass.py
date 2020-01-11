from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('./datasets/iris.csv', names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

sns.pairplot(df, hue='species')
plt.show()

ds = df.values
X = ds[:,0:4].astype(float)
Y_obj = ds[:,4]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = to_categorical(Y).astype(int)

model = Sequential([Dense(16, input_dim=4, activation='relu'), Dense(3, activation='softmax')])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y_encoded, epochs=50, batch_size=1)

print(model.evaluate(X,Y_encoded))