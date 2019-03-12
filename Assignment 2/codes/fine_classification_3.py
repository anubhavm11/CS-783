import numpy as np
import cv2
import os
import sys

from tqdm import tqdm
import keras
from classification_models.resnet import ResNet18, preprocess_input
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

rootdir = "../dataset/train/full_data_original/"

y = []
y2 = []
x=[]
C = 0
label2 = 0

classDict={0:'birds',1:'cars',2:'aircrafts',3:'dogs',4:'flowers'}

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
			img2 = cv2.imread(image_path,1)

			img2 = cv2.resize(img2, (224, 224))
			# img2 = np.expand_dims(img2, axis=0)
			x.append(img2)
			for i,j in classDict.items():
				if(j==subdir):
					y.append(i)
					break
			y2.append(int(subsubdir)-1)
			# if(cnt>=5):
			# 	cnt=0
			# 	break				
		label +=1
	label2+=1

# y = np.eye(n_classes)[np.asarray(y)]

# x,y,y2=shuffle(x,y,y2)

X=np.asarray(x)
y=np.asarray(y)
y2=np.asarray(y2)
# print(y)
# print(y2)

# X = preprocess_input(X)

from keras.models import load_model

model = load_model("../working/FineClass.hdf5")
extract = keras.models.Model(model.inputs, model.layers[-2].output)
X = extract.predict(X)
print(X.shape)

import pickle
clf = pickle.load(open("../working/coarseBest.pickle", "rb"))
y_pred = clf.predict(X)
# from sklearn.metrics import accuracy_score
# score = accuracy_score(y, y_pred)
# print("accuracy on test data is : " + str(score))	

modelArr=[]

clf2 = pickle.load(open("../working/flowers.pickle", "rb"))
y_pred3 = clf2.predict(X)
for i in range(0,5):
	modelArr.append(pickle.load(open("../working/"+classDict[i]+".pickle", "rb")))

k=0
s=0
s2=0
y_pred2=np.zeros(y.shape[0])
for i in tqdm(range(X.shape[0])):
	k+=1
	y_pred2[i]=(modelArr[y[i]].predict([X[i]]))[0]
	if(y_pred[i]==y[i]):		
		if(y_pred2[i]==y2[i]):
			s+=1
print(s)
print(k)
# print(y)
# print(y_pred)
# print(y2)
# print(y_pred2)
# print(y_pred3)