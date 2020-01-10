import pandas as pd
df = pd.read_csv('./datasets/pid.csv', names=["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])

# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(12,12))
# sns.heatmap(df.corr(), linewidth=0.1, vmax=0.5, cmap="Blues", linecolor="white", annot=True)
# plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

seed=0
np.random.seed(seed)
tf.set_random_seed(seed)

ds = df.values
X = ds[:,0:8]
Y = ds[:,8]

model = Sequential([Dense(12, input_dim=8, activation='relu'), Dense(8, activation='relu'), Dense(1, activation='sigmoid')])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=200, batch_size=10)

print(model.evaluate(X,Y))