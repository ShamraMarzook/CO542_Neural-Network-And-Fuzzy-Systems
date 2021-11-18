# E/16/232

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [0, 1, 1, 0]

plt1=plt.scatter([point[0] for point in data], [point[1] for point in data], c = labels)
plt.legend(*plt1.legend_elements())
plt.title('XOR')
plt.show()

model = MLPClassifier(hidden_layer_sizes=(5),max_iter=10000)

netXOR = model.fit(data, labels)
y_pred = netXOR.predict([[0,0],[0,1],[1,0],[1,1]])
print(y_pred)

