import numpy as np
import cv2
import os
import sys

from tqdm import tqdm
import keras
from keras.models import load_model

model = load_model("../working/FineClass.hdf5")
extract = keras.models.Model(model.inputs, model.layers[-2].output)

import pickle
clf = pickle.load(open("../working/coarseBest.pickle", "rb"))

rootdir = "../test/"

img_name = []

classDict={0:'birds',1:'cars',2:'aircrafts',3:'dogs',4:'flowers'}

modelArr=[]

for i in range(0,5):
	modelArr.append(pickle.load(open("../working/"+classDict[i]+".pickle", "rb")))


fp = open('../results/out.txt', 'w')

for img in tqdm(os.listdir(rootdir)):

	image_path = rootdir + img
	img2 = cv2.imread(image_path,1)
	img2 = cv2.resize(img2, (224, 224))

	X = (extract.predict(np.array([img2])))
	y_pred = clf.predict(X)[0]
	y_pred2=(modelArr[y_pred].predict(X))[0]
	fp.write(img+' '+classDict[y_pred]+' '+classDict[y_pred]+'@'+str(y_pred2+1)+'\n')

fp.close()
