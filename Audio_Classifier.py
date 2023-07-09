import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation

mfcc=128
audio_dataset_path='Speech/'
metadata=pd.read_csv('Speech_Data.csv')
metadata.head()

def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=mfcc)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

extracted_features=[]

for i in range(2,436):
    file_name = metadata['File_name'][i]
    final_class_labels=metadata['Labels'][i]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])

for i in range(436,872):
    file_name = metadata['File_name'][i]
    final_class_labels=metadata['Labels'][i]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])

extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head(10)

X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=Sequential()
model.add(Dense(128,input_shape=(mfcc,)))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')
# hist = model.fit(X, y, epochs=num_epochs, verbose=1) This is for training
model.load_weights("AU_54.h5")

from matplotlib import pyplot

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

from sklearn.metrics import confusion_matrix,classification_report

y_pred=model.predict(X_train,verbose=1)
yp = []
yt = []
for i in range(696):
    x = np.argmax(y_pred[i])
    yp.append(x)
    x = np.argmax(y_train[i])
    yt.append(x)

print('\nTrain-Confusion Matrix')
cft = confusion_matrix(yt, yp)
print(cft)

print('\nTrain-Classification Report')
target_names = ['Depressed', 'Not Depressed']
print(classification_report(y_true=yt, y_pred=yp, target_names=target_names))


y_pred=model.predict(X_test,verbose=0)
yp = []
yt = []
for i in range(174):
    x = np.argmax(y_pred[i])
    yp.append(x)
    x = np.argmax(y_test[i])
    yt.append(x)

print('\nValidation-Confusion Matrix')
cf = confusion_matrix(yt, yp)
print(cf)

print('\nValidation-Classification Report')
target_names = ['Depressed', 'Not Depressed']
print(classification_report(y_true=yt, y_pred=yp, target_names=target_names))


#Testing with custom audio files
a,b=librosa.load("file_name", res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=a, sr=b, n_mfcc=mfcc)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
x=[]
x.append([mfccs_scaled_features])
xdf=pd.DataFrame(x,columns=['feature'])
x=np.array(xdf['feature'].tolist())
y1=model.predict(x)
print(y1[0][0])