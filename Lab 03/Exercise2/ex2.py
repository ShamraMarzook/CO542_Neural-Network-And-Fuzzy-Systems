#E/16/232

from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import time
import numpy as np
from mlxtend.plotting import plot_decision_regions

data = [[0.1, 0.1], [0.8, 0.3], [0.3, 0.4], [0.6, 0.9]]
labels = [0, 1, 0, 1]

# Plot these vectors (Use a scatter plot)
plt.scatter([point[0] for point in data], [point[1] for point in data], c=labels)
plt.show()

# Building a perceptron
classifier = Perceptron(max_iter=40)
# Observe the training time
start_time = time.time()
classifier.fit(data, labels)
stop_time = time.time()

training_time = stop_time - start_time

print('Training time = ', training_time)
print('No. of iterations = ', classifier.n_iter_)

# test input
x_test = [[0.5, 0.7]]
prediction = classifier.predict(x_test)
print('Prediction: ', prediction)

data = np.array(data)
labels = np.array(labels)
plot_decision_regions(data, labels, clf=classifier, legend=2)       # Plotting the classification line
# Adding the test input to the plot
plt.scatter([point[0] for point in x_test], [point[1] for point in x_test], label='test input', c='g')
plt.axis([0, 0.9, 0, 1])                # Zoom in the plot
plt.show()
