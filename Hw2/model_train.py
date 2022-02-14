# from google.colab import drive
# drive.mount('/content/drive')

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf

from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model

%load_ext tensorboard
!rm -rf ./logs/ 

FREEZE_LAYERS = 2

train_datagen = ImageDataGenerator()

train_batches = train_datagen.flow_from_directory('/content/drive/MyDrive/sample/Train/', target_size=(224, 224), interpolation='bicubic', class_mode='categorical', shuffle=True, batch_size=8)
        
valid_datagen = ImageDataGenerator()

valid_batches = valid_datagen.flow_from_directory('/content/drive/MyDrive/sample/Test/', target_size=(224, 224), interpolation='bicubic', class_mode='categorical', shuffle=False, batch_size=8)

for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

net = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3))
x = net.output
x = Flatten()(x)
output_layer = Dense(2, activation='softmax', name='softmax')(x)

net_final  = Model(inputs=net.input, outputs=output_layer)

for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

net_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

print(net_final.summary())

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // 256,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // 256,
                        epochs = 10,callbacks=[tensorboard_callback])


%tensorboard --logdir logs/fit