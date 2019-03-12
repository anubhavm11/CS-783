import keras
from classification_models.resnet import ResNet18, preprocess_input
import matplotlib.pyplot as plt 
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

# prepare your data
X = pickle.load(open("x_total.pickle", "rb"))
y = pickle.load(open("y_total.pickle", "rb"))

n_classes = 36
y = np.eye(n_classes)[np.asarray(y)]

X=np.asarray(X)

X = preprocess_input(X)

X, y = shuffle(X, y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)


# build model
base_model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])
model.summary()

# train
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])



BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100

NUM_EPOCHS = 5
EARLY_STOP_PATIENCE = 3

STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath = '../working/FineClass.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')

fit_history = model.fit(
        X_train,y_train,
        epochs = NUM_EPOCHS,
        validation_data = (X_test,y_test),
        callbacks=[cb_checkpointer, cb_early_stopper]
)

# model.load_weights("../working/coarseBest.hdf5")

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


# model.save("../working/FineClass.hdf5")

# y_pred = model.predict(X_test)

# print("Checking accuracy")
# y_pred = np.argmax(y_pred, axis=1)
# y_true = np.argmax(y_test, axis=1)
# from sklearn.metrics import accuracy_score
# score = accuracy_score(y_true, y_pred)
# print("accuracy on test data is : " + str(score))