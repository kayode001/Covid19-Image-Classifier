# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from skimage.transform import resize   
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import skimage
import sys
import cv2 as cv
plt.style.use('fivethirtyeight')
 

#load data
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
image_data = np.load('images.npy')
labels= pd.read_csv('labels.csv', delimiter=',').values

x_train = np.load('trainimage.npy')
x_test = np.load('testimage.npy')
y_test = pd.read_csv('testLabels.csv', delimiter=',').values
y_train = pd.read_csv('trainLabels.csv', delimiter=',').values




# apply Gaussian blur
x_test = cv.blur(x_test,(5,5))
x_train = cv.blur(x_train,(5,5))


#print count and uniques values
test_label_unique, test_label_counts = np.unique(y_test, return_counts=True)
train_label_unique, train_label_counts = np.unique(y_train, return_counts=True)


#shape of data
print("shape of X_TRAIN: ",x_train.shape)
print("shape of X_TRAIN: ",y_train.shape)

#normalize the pixels
x_train = x_train / 255
x_test = x_test / 255
 

#converting label to one hot
y_train_one_hot = pd.get_dummies(y_train.flatten())
y_test_one_hot = pd.get_dummies(y_test.flatten())

print(y_test_one_hot)

y_train_labels_one_hot = y_train_one_hot.columns
y_test_labels_one_hot = y_test_one_hot.columns
print(y_train_labels_one_hot[0])



model = Sequential()

#adding layers

#convolution  layer
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(128,128,3)))

#pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#flatten the image(reduce to 1d array)
model.add(Flatten())


model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(250, activation='relu'))
model.add(Dense(12, activation='softmax'))


#compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#fit our data to the model we just designed
fit_model = model.fit(x_train, y_train_one_hot,
           batch_size=32, epochs=1, validation_split=0.3 )


#evaluate our model
print(model.evaluate(x_train, y_train_one_hot)[1])





#make predictions
#load images needed according to step 6 instruction visualize predictions for x_test[2], x_test[3], x_test[33], x_test[36], x_test[59].
new_image =  x_test[2]


img = plt.imshow(new_image)

from skimage.transform import resize
resized_image = resize(new_image, (128,128,3))

#show image to be predicted
img = plt.imshow(resized_image)

predictions = model.predict(np.array( [resized_image] ))


print(predictions)

#get predicted index and value
predicted_index = np.argmax(predictions,axis=-1)[0]
prediction_class = y_test_labels_one_hot[predicted_index]
print(prediction_class)



