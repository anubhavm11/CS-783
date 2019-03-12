import cv2
import numpy as np
import os
import pandas as pd
import csv
import glob2
import pickle
import matplotlib.pyplot as plt 

from sklearn.cluster import MiniBatchKMeans
# from sklearn.neural_network import MLPClassifier
from keras.optimizers import SGD
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

orb = cv2.ORB_create()
folders = glob2.glob("../dataset/train/*")

folders2 = glob2.glob("../dataset/valid/*")

# imagenames__list = []
train_x = []
train_y = []
train_y_labels = []
a=0
# print(folders)
dico = []																																																																																	

numImages=0

NUM_CLASS = 16
from tqdm import tqdm

for folder in tqdm(folders):
	for f in glob2.glob(folder+'/*.jpg'):
		numImages+=1
		img = cv2.imread(f)
		# train_x.append(img)

		kp = orb.detect(img,None)
		kp, des = orb.compute(img, kp)
		for d in des:
			dico.append(d)

		tmp=[0]*16
		tmp[a]=1
		train_y.append(tmp)
		train_y_labels.append(a)
	a+=1
        
# train_x=np.array(train_x)
# train_y=np.array(train_y)

k = NUM_CLASS * 10

batch_size = numImages * 3
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(dico)
pickle_out = open("kmeans.pickle","wb")
pickle.dump(kmeans, pickle_out)
pickle_out.close()

kmeans.verbose = False

histo_list = []

for folder in tqdm(folders):
	for f in glob2.glob(folder+'/*.jpg'):
		# numImages+=1
		img = cv2.imread(f)
		# train_x.append(img)

		kp = orb.detect(img,None)
		kp, des = orb.compute(img, kp)

		histo = np.zeros(k)
		nkp = np.size(kp)

		for d in des:
			idx = kmeans.predict([d])
			histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly

		histo_list.append(histo)

valid_y = []
histo_list2 = []

a=0

for folder in tqdm(folders2):
	for f in glob2.glob(folder+'/*.jpg'):
		# numImages+=1
		img = cv2.imread(f)
		# train_x.append(img)

		kp = orb.detect(img,None)
		kp, des = orb.compute(img, kp)

		histo = np.zeros(k)
		nkp = np.size(kp)

		for d in des:
			idx = kmeans.predict([d])
			histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly

		histo_list2.append(histo)
		tmp=[0]*16
		tmp[a]=1
		valid_y.append(tmp)
	a+=1x

# for img in train_x:
#     kp = orb.detect(img,None)
#     kp, des = orb.compute(img, kp)

#     histo = np.zeros(k)
#     nkp = np.size(kp)

#     for d in des:
#         idx = kmeans.predict([d])
#         histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly

#     histo_list.append(histo)


histo_list=np.asarray(histo_list)
train_y=np.asarray(train_y)
train_y_labels=np.asarray(train_y_labels)

np.save("BagOfWords_Features.npy",histo_list)
np.save("BagOfWords_LabelVectors.npy",train_y)
np.save("BagOfWords_Labels.npy",train_y_labels)

model = Sequential()
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(train_y.shape[1], activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,metrics=['accuracy'])

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("../working/BagOfWords.hdf5", monitor='sacc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=10)
# early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

fit_history=model.fit(histo_list, train_y, validation_split=0.1, validation_steps=10, epochs=200,steps_per_epoch=100,verbose=2,callbacks = [checkpoint])

plt.figure(1, figsize = (15,8)) 
    
plt.subplot(221)  
plt.plot(fit_history.history['acc'])  
plt.plot(fit_history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
    
plt.subplot(222)  
plt.plot(fit_history.history['loss'])  
plt.plot(fit_history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()

