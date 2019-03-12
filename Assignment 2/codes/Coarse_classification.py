import keras
from classification_models.resnet import ResNet18, preprocess_input
import matplotlib.pyplot as plt 
# prepare your data
# X = ...
# y = ...

# X = preprocess_input(X)

from keras.preprocessing.image import ImageDataGenerator


image_height = 155
image_width = 225

BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100

NUM_EPOCHS = 20
EARLY_STOP_PATIENCE = 3

STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.2)

# flow_From_directory generates batches of augmented data (where augmentation can be color conversion, etc)
# Both train & valid folders must have NUM_CLASSES sub-folders
train_generator = data_generator.flow_from_directory(
        '../dataset/train/full_data',
        target_size=(image_height, image_width),
        batch_size=BATCH_SIZE_TRAINING,
        class_mode='categorical',
        subset='training')

validation_generator = data_generator.flow_from_directory(
        '../dataset/train/full_data',
        target_size=(image_height, image_width),
        batch_size=BATCH_SIZE_VALIDATION,
        class_mode='categorical',
        subset='validation') 



n_classes = 5

# build model
base_model = ResNet18(input_shape=(image_height,image_width,3), weights='imagenet', include_top=False)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])

# print(model.layers[86])
for i in range(0,86):
	model.layers[i].trainable = False

model.summary()
# print(model.layers[1].trainable)
# train
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X, y)

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath = '../working/coarseBest.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')

fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
        epochs = NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=STEPS_PER_EPOCH_VALIDATION,
        callbacks=[cb_checkpointer, cb_early_stopper]
)

model.load_weights("../working/coarseBest.hdf5")

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

