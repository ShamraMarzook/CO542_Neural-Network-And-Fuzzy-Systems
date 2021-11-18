# E/16/232

from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# Creating a variable named data that is a list that contains the four possible inputs to an OR gate.
data = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [0, 1, 1, 1]

# plt.scatter([point[0] for point in data], [point[1] for point in data], c=labels)
# The third parameter "c = labels" will make the points with label 1 a different color than points with label 0.
# plt.show()

# Building a perceptron to learn OR.
classifier = Perceptron(max_iter=40)
classifier.fit(data, labels)
print(classifier.score(data, labels))

x_test = np.arange(0, 1, 0.009)
y_test = np.arange(0, 1, 0.009)

X, Y = np.meshgrid(x_test, y_test)
z_test = np.array([X, Y]).T.reshape(-1, 2).tolist()

predictions = classifier.predict(z_test)
predictions = predictions.reshape(x_test.size, y_test.size)
print('Prediction:')
print(predictions)

# Plot the surface.
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, predictions, cmap='plasma', rstride=1, cstride=1)
ax.set_title('Surface plot of OR Gate')
ax.set_xlabel('Input - 1')
ax.set_ylabel('Input - 2')
plt.show()
