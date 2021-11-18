# E/16/232

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LeakyReLU
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

# 1. Import the CIFER-10 data set using keras.datasets.
from keras.datasets import cifar10


# 2. Study the shapes of the training and testing datasets.
#download CIFER-10 data and split into train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('Shape of training dataset: X_train = ', X_train.shape, '| Y_train = ', y_train.shape)
print('Shape of testing dataset: X_test = ', X_test.shape, '| Y_test = ', y_test.shape)


# 3. Visualize some images in the train and test tests to understand the dataset. You may use
# matplotlib.pyplot.imshow to display the images in a grid.
fig, axes1 = plt.subplots(5, 5, figsize=(5, 5))
for j in range(5):
    for k in range(5):
        i = np.random.choice(range(len(X_train)))
        axes1[j][k].set_axis_off()
        axes1[j][k].imshow(X_train[i:i+1][0])
plt.show()

fig, axes2 = plt.subplots(5, 5, figsize=(5, 5))
for j in range(5):
    for k in range(5):
        i = np.random.choice(range(len(X_test)))
        axes2[j][k].set_axis_off()
        axes2[j][k].imshow(X_test[i:i+1][0])
plt.show()


# 4. Under the data pre-processing procedures,
# • Reshape the input datasets accordingly.
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)

# • Normalize the pixel values in a range between 0 to 1.
X_train_normalized = X_train/255
X_test_normalized = X_test/255

# • Convert the class labels into One-Hot encoding vector. Clearly mention the requirement of this conversion.
ohe = OneHotEncoder(sparse=False)
ohe.fit(y_train)
y_train = ohe.transform(y_train)
y_test = ohe.transform(y_test)

# • Use sklearn.model selection.train test split to further split the training dataset into validation
# and training data (e.g. allocate 0.2 of the training set as validation data).
train_X, valid_X, train_label, valid_label = train_test_split(X_train, y_train, test_size=0.2, random_state=13)


# 5. Build the CNN model with three convolutional layers followed by a dense layer and an output layer
# accordingly. In this case,
# • Select 3 X 3 as the kernal size of each filter.
# • Use different number of filters in each convolutional layer (e.g. first layer 32 filters, second
# layer 64 filters, third layer 128 filters).
# • Use LeakyReLU as the activation function. Mention the advantage of using LeakyReLU over
# ReLU activation function.
# • Use 2 X 2 MaxPooling layers, and Dropout layers according to the requirements and mention
# the purpose behind the usage of Dropout Layers.
#create model
model = Sequential()
#add model layers
model.add(Conv2D(32, kernel_size=(3, 3), activation=LeakyReLU(alpha=0.1), input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation=LeakyReLU(alpha=0.1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation=LeakyReLU(alpha=0.1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))


# 6. Compile the model using appropriate parameters and generate the model summery using
# model.summary() function (In this case make sure to specify the metrics as accuracy).
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
summary = model.summary()
print(summary)


# 7. Train the compiled model using model.fit function and observe the train and validation set performances.
# In this case, you may have to select an appropriate number of epochs (e.g. 25) and
# batch size (e.g. 64, 128 or 256).
model_hostory = model.fit(train_X, train_label, validation_data=(valid_X, valid_label), epochs=35, batch_size=64)

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# summarize history for accuracy
axs[0].plot(model_hostory.history['accuracy'])
axs[0].plot(model_hostory.history['val_accuracy'])
axs[0].set_title('Model Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend(['train', 'validate'], loc='upper left')

# summarize history for loss
axs[1].plot(model_hostory.history['loss'])
axs[1].plot(model_hostory.history['val_loss'])
axs[1].set_title('Model Loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].legend(['train', 'validate'], loc='upper right')
plt.show()


# 8. Evaluate the model performance using test set. Identify the test loss and test accuracy.
test_eval = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", test_eval[0])
print("Test accuracy:", test_eval[1])


# 9. Use the trained model to make predictions for the test data and visualize the model performance
# under each class using sklearn.metrics.classification report.
predictions = model.predict(X_test)
predictions = ohe.inverse_transform(predictions)
y_test = ohe.inverse_transform(y_test)
print('Prediction:')
print(predictions)
print('y_test:')
print(y_test)

target_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('classification_report:')
print(classification_report(y_test, predictions, target_names=target_names))
