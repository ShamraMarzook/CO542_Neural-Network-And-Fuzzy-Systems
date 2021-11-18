# E/16232

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


# 1. Load the dataset as a Panda's dataframe and drop any unnecessary columns.
iris_data = pd.read_csv('iris.csv')
iris_data = iris_data.drop(columns=['Id'])


# 2. Visualize the relationships between dataset features using seaborn.pairplot
# sb.set_style("ticks")
sb.pairplot(iris_data, hue='Species', diag_kind="auto", kind="scatter", palette="husl")
plt.show()


# 3. Separate the dataset into features and labels.
x_col = [col for col in iris_data.columns if col not in ['Species']]
y_col = ['Species']
features = iris_data[x_col]
label = iris_data[y_col]
# print(label)


# 4. Convert categorical data in your dependent variable into numerical values using Sklearn Label Encoder.
label_copy = label.copy()
lb = LabelEncoder()
label_copy['Species'] = lb.fit_transform(label['Species'])
# print(label['Species'].unique())
# print(label_copy['Species'].unique())


# 5. Split the data set as train and test data accordingly (E.g.: 80% training data, 20% test data).
X_train, X_test, y_train, y_test = train_test_split(features, label_copy, random_state=1, test_size=0.2)


# 6. Scale the independent train and test datasets using Sklearn Standard Scaler.
sc_X = StandardScaler()
X_trainscaled = sc_X.fit_transform(X_train)
X_testscaled = sc_X.transform(X_test)


# 7. Model the MLP using MLPClassifier.
model = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), activation="relu", random_state=1)
# model = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), activation="relu", random_state=1, max_iter=100)
# model = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), activation="relu", random_state=1, max_iter=300)
# model = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), activation="relu", random_state=1, max_iter=500)


# 8. Make predictions using previously reserved test data set.
clf = model.fit(X_trainscaled, y_train.values.ravel())
y_pred = clf.predict(X_testscaled)
# print(y_pred)


# 9. Observe the accuracy of the model by plotting the confusion metrix.
print('Accuracy: ', clf.score(X_testscaled, y_test))
fig = plot_confusion_matrix(clf, X_testscaled, y_test, display_labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()


# 10. Generate the classification report using Sklearn Classification Report and comment on each of the value you obtained in that report.
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)


# 11. Repeat the task with 100,300,500 iterations and comment on your results.
# 12. Repeat the task with varying test and train data set sizes and comment on your observations.
# 13. Repeat the task with dierent learning rates: 0.002, 0.5, 1.00 and comment on the results obtained.