import numpy as np
import os
import os.path
import sys
import pickle
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import cv2
import sys
from sklearn.neighbors import KNeighborsClassifier
import glob2

def cosine(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    return np.dot(v1, v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))

ImageName=np.load('ImageName.npy')
Features=np.load('BagOfWords_Features.npy')

pickle_kmeans = open("kmeans.pickle","rb")
kmeans = pickle.load(pickle_kmeans)

NUM_CLASS=16
k = NUM_CLASS * 10


test_images=glob2.glob('../dataset/sample_test/test/*')

orb = cv2.ORB_create()

for f in test_images:
	print(f)
	img = cv2.imread(f)

	kp = orb.detect(img,None)
	kp, des = orb.compute(img, kp)

	histo = np.zeros(k)
	nkp = np.size(kp)

	for d in des:
		idx = kmeans.predict([d])
		histo[idx] += 1/nkp

	#use cosine distance
	tmp=[]
	for v in Features:
		tmp.append(cosine(v,histo))
	tmp=np.asarray(tmp)
	pred = np.argsort(-tmp)

	pred = ImageName[pred]
	f2 = '../results/' + f.split('/')[-1].split('.jpg')[0] + '.txt'
	fp = open(f2, 'w')
	for fname in pred:
		fp.write(fname + '\n')
	fp.close()
