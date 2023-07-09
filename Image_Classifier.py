import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from tensorflow.keras import regularizers
from keras.layers import Conv2D, AvgPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from matplotlib import pyplot

#train_dir = "./Image/train/"
#val_dir = "./Image/val/"
#data = ImageDataGenerator()
#train = data.flow_from_directory(train_dir,target_size=(48,48),class_mode='categorical',batch_size=100,shuffle=False)
#validation = data.flow_from_directory(val_dir,target_size=(48,48),class_mode='categorical',batch_size=100,shuffle=False)
label_map = (train.class_indices)


model = Sequential()
model.add(InputLayer(input_shape=(48, 48, 3)))
model.add(Conv2D(32, kernel_size=2,padding='same'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(optimizer="adam",loss='binary_crossentropy', metrics=['accuracy'])

# hist= model.fit(train,epochs=15,verbose=1,validation_data=validation) This is for training 
# print(hist.history.keys())
model.load_weights("IM_66_67_7_9.h5")

from sklearn.metrics import classification_report, confusion_matrix

Y_pred = model.predict(train)
y_pred = np.argmax(Y_pred, axis=1)
print('Train-Confusion Matrix')
print(confusion_matrix(train.classes, y_pred))
print('\nTrain-Classification Report')
target_names = ['Depressed', 'Not Depressed']
print(classification_report(train.classes, y_pred, target_names=target_names))
print("\n")

Y_pred = model.predict(validation)
y_pred = np.argmax(Y_pred, axis=1)
print('Validation-Confusion Matrix')
print(confusion_matrix(validation.classes, y_pred))
print('\nValidation-Classification Report')
target_names = ['Depressed', 'Not Depressed']
print(classification_report(validation.classes, y_pred, target_names=target_names))

pyplot.figure(figsize=(15,5))
pyplot.subplot(1, 2, 1)
pyplot.plot(hist.history['loss'], 'r', label='Training loss')
pyplot.plot(hist.history['val_loss'], 'g', label='Validation loss')
pyplot.legend()
pyplot.subplot(1, 2, 2)
pyplot.plot(hist.history['accuracy'], 'r', label='Training accuracy')
pyplot.plot(hist.history['val_accuracy'], 'g', label='Validation accuracy')
pyplot.legend()
pyplot.show()

#This is for prediction
path = "image.jpg"
img = image.load_img(path, target_size=(48, 48))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = np.vstack([x])
cl = model.predict(x)
print("Image prediction->",np.argmax(cl)) 
print(cl)