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

# imagenames__list = []
a=0
# print(folders)
dico = []																																																																																	

numImages=0

NUM_CLASS = 5
from tqdm import tqdm

y = []
y2 = []
x=[]
C = 0
label2 = 0

classDict={0:'birds',1:'cars',2:'aircrafts',3:'dogs',4:'flowers'}

dico=[]
rootdir = "../dataset/train/full_data_original/"

for subdir in tqdm(os.listdir(rootdir)):
	label = 0
	# print(subdir)
	for subsubdir in tqdm(os.listdir(rootdir + subdir)):
		cnt = 0
		# print(subsubdir)
		cnt2=0
		for img in os.listdir(rootdir + subdir + '/' + subsubdir + '/'):
			cnt += 1

			image_path = rootdir + subdir + '/' + subsubdir + '/' + img
			# print(image_path)
			img2 = cv2.imread(image_path)

			img2 = cv2.resize(img2, (224, 224))

			x.append(img2)

			kp = orb.detect(img2,None)
			kp, des = orb.compute(img2, kp)
			for d in des:
				dico.append(d)

			for i,j in classDict.items():
				if(j==subdir):
					y.append(i)
					break
			y2.append(int(subsubdir)-1)
						
		label +=1
	label2+=1



k = NUM_CLASS * 10

kmeans = MiniBatchKMeans(n_clusters=k, verbose=1).fit(dico)
pickle_out = open("kmeans.pickle","wb")
pickle.dump(kmeans, pickle_out)
pickle_out.close()

kmeans.verbose = False

histo_list = []

for img2 in x:
	kp = orb.detect(img2,None)
	kp, des = orb.compute(img2, kp)
	
	histo = np.zeros(k)
	nkp = np.size(kp)

	for d in des:
		idx = kmeans.predict([d])
		histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly

	histo_list.append(histo)


a=0


histo_list=np.asarray(histo_list)
y=np.asarray(y)
y=np.eye(5)[y]
np.save("BagOfWords_Features.npy",histo_list)
np.save("BagOfWords_LabelVectors.npy",y)

model = Sequential()

model.add(Dense(5,input_dim=k, activation = 'softmax'))
# model.add(Dense(n_classes, activation = 'softmax'))

model.summary()
# print(model.layers[1].trainable)
# train
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X, y)

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = 3)
cb_checkpointer = ModelCheckpoint(filepath = '../working/bagOfWords.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')

fit_history = model.fit(
        histo_list,y,
        validation_split=0.1, validation_steps=10, epochs=200,steps_per_epoch=100,verbose=2,
        callbacks=[cb_checkpointer, cb_early_stopper]
)

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

