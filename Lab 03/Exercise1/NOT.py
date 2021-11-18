# E/16/232

from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# Creating a variable named data that is a list that contains
# the four possible inputs to an NOT gate.
data = [[0], [1]]
labels = [1, 0]

# Building a perceptron to learn NOT.
classifier = Perceptron(max_iter=40)
classifier.fit(data, labels)
print(classifier.score(data, labels))

x_test = np.arange(0, 1, 0.009)
x_test = x_test.reshape(-1, 1)

predictions = classifier.predict(x_test)
print('Prediction:')
print(predictions)

# Plot the surface.
plt.plot(x_test, predictions)
plt.title('Surface plot of NOT Gate')
plt.xlabel('Input')
plt.ylabel('Prediction')
plt.show()
