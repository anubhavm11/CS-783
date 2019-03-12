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

rootdir = "../dataset/train/full_data_original/dogs/"

label = 0
y = []
x=[]

for subdir in tqdm(os.listdir(rootdir)):
	for img in os.listdir(rootdir + subdir + '/'):
		image_path = rootdir + subdir + '/' +  img
		# print(image_path)
		img2 = cv2.imread(image_path,1)

		# cv2.imshow('image',img2)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		# sys.exit(1)

		img2 = cv2.resize(img2, (224, 224))
		# img2 = np.expand_dims(img2, axis=0)
		x.append(img2)
		y.append(int(subdir)-1)
	label +=1
n_classes=label

image_height = 224
image_width = 224

NUM_EPOCHS = 20
EARLY_STOP_PATIENCE = 3


y = np.asarray(y)

X=np.asarray(x)

X = preprocess_input(X)

X, y = shuffle(X, y, random_state=0)

from keras.models import load_model

model = load_model("../working/FineClass.hdf5")
extract = keras.models.Model(model.inputs, model.layers[-2].output)
X = extract.predict(X)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense

# model = Sequential()

# model.add(Dense(n_classes,input_dim=512, activation = 'softmax'))
# # model.add(Dense(n_classes, activation = 'softmax'))

# model.summary()
# # print(model.layers[1].trainable)
# # train
# model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
# # model.fit(X, y)

# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
# cb_checkpointer = ModelCheckpoint(filepath = '../working/flowers.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')

# fit_history = model.fit(
#         X_train,y_train,
#         epochs = 50,
#         validation_data = (X_test,y_test),
#         callbacks=[cb_checkpointer, cb_early_stopper]
# )

from sklearn.svm import SVC
clf = SVC(gamma = "auto", verbose=True, max_iter=200)
print("fitting SVM")
clf.fit(X_train, y_train)
print("getting accuracy SVM")

y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print("accuracy on test data is : " + str(score))

import pickle

pickle_out = open("../working/dogs.pickle","wb")
pickle.dump(clf, pickle_out)
pickle_out.close()

# plt.figure(1, figsize = (15,8)) 
    
# plt.subplot(221)  
# plt.plot(fit_history.history['acc'])  
# plt.plot(fit_history.history['val_acc'])  
# plt.title('model accuracy')  
# plt.ylabel('accuracy')  
# plt.xlabel('epoch')  
# plt.legend(['train', 'valid']) 
    
# plt.subplot(222)  
# plt.plot(fit_history.history['loss'])  
# plt.plot(fit_history.history['val_loss'])  
# plt.title('model loss')  
# plt.ylabel('loss')  
# plt.xlabel('epoch')  
# plt.legend(['train', 'valid']) 

# plt.show()




