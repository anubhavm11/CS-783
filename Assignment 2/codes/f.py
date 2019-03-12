import numpy as np
import cv2
# from matplotlib import pyplot as plt
import os
import sys
# import shutil

from tqdm import tqdm
# from skimage.io import imread

rootdir = "../dataset/train/full_data_original/"

label = 0
y = []
y2 = []
x=[]
C = 0
label2 = 0
for subdir in tqdm(os.listdir(rootdir)):
	
	for subsubdir in tqdm(os.listdir(rootdir + subdir)):
		cnt = 0
		# print(subsubdir)
		
		for img in os.listdir(rootdir + subdir + '/' + subsubdir + '/'):
			cnt += 1
			image_path = rootdir + subdir + '/' + subsubdir + '/' + img
			# print(image_path)
			img2 = cv2.imread(image_path,1)

			# cv2.imshow('image',img2)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			# sys.exit(1)

			img2 = cv2.resize(img2, (224, 224))
			# img2 = np.expand_dims(img2, axis=0)
			x.append(img2)
			y.append(label)
			y2.append(label2)
		label +=1

			# shutil.copy(rootdir + subdir + '/' + subsubdir + '/' + img, "./full_data/" + subdir + '/' + img)

		print("found " + str(cnt) + " images in " + subdir + "/" + subsubdir + "\n" + 20*'#')
	label2+=1

# print("final shape of X : " + str(x.shape))
# print("final length of y : " + str(len(y)))
# print("max label of y : " + str(max(y)))
# print("final length of y2 : " + str(len(y2)))
# print("max label of y2 : " + str(max(y2)))

import pickle

pickle_out = open("x_total.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y_total.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_out = open("y_total_2.pickle","wb")
pickle.dump(y2, pickle_out)
pickle_out.close()

