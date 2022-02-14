import sys
import cv2
import numpy
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX_normalize=trainX.astype('float32')/255.0
testX_normalize=testX.astype('float32')/255.0

trainY_onehot=np_utils.to_categorical(trainY)
testY_onehot=np_utils.to_categorical(testY)

model=Sequential([
  Conv2D(64, (3, 3), input_shape = (32, 32, 3), padding='same',activation='relu'),
  Conv2D(64, (3, 3), activation='relu', padding='same'),
  MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  Conv2D(128, (3, 3), activation='relu', padding='same'),
  Conv2D(128, (3, 3), activation='relu', padding='same',),
  MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  Conv2D(256, (3, 3), activation='relu', padding='same',),
  Conv2D(256, (3, 3), activation='relu', padding='same',),
  Conv2D(256, (3, 3), activation='relu', padding='same',),
  MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  Conv2D(512, (3, 3), activation='relu', padding='same',),
  Conv2D(512, (3, 3), activation='relu', padding='same',),
  Conv2D(512, (3, 3), activation='relu', padding='same',),
  MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  Conv2D(512, (3, 3), activation='relu', padding='same',),
  Conv2D(512, (3, 3), activation='relu', padding='same',),
  Conv2D(512, (3, 3), activation='relu', padding='same',),
  MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  Flatten(),
  Dense(4096, activation='relu'),
  #Dropout(0.5),
  Dense(4096, activation='relu'),
  #Dropout(0.5),
  Dense(10, activation='softmax')
])

#print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.01,momentum=0.9),metrics=['accuracy'])
acc = model.fit(trainX_normalize,trainY_onehot,batch_size=128,epochs=20)
#print(acc.history['accuracy'])
#print(acc.history['loss'])
#result = model.evaluate(testX_normalize,testY_onehot,batch_size=10000)
#print('Test Acc: ',result[1])

#print(acc.history['accuracy'])
a = numpy.array(acc.history['accuracy'])
b = range(1, 21)

fig = plt.figure()
plt.title('Accuracy_rate')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xlim([0,20])
plt.ylim([0,1])
plt.xticks(range(0, 21))
plt.plot(b, a, '--')
fig.savefig('Accuracy_rate.jpg')

a = numpy.array(acc.history['loss'])
b = range(1, 21)

fig_loss = plt.figure()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xlim([0,20])
plt.ylim([0,3])
plt.xticks(range(0, 21))
plt.plot(b, a, '--')
fig_loss.savefig('loss.jpg')

model.save('my_model.h5')