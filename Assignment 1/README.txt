We are submitting the codes of the different models we tried, and the resnet model is our main submission.
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Place the test images in the path "Assignment\ 1/dataset/sample_test/test/".

All codes are present in the Assignment\ 1/codes/ folder, and should be run from that folder.
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
For running the resnet model, 

python tl_test.py

Assignment\ 1/working/best.hdf5 stores weights for the neural net, and Assignment\ 1/input/resnet50 contains the parameters for the net.
Image_Name_resnet.npy contains names of images for each class for creating output files.
transfer-learning-implementation_2.py was used for training the model.
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
For running the bag of words with ORB descriptors cosine model, 

python bag-of-words-cosine.py

BagOfWords_Features.npy stores embedding for the training images, BagOfWords_Labels.npy store corresponding labels.
kmeans.pickle stores the k cluster centers.
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
For running the bag of words with ORB descriptors neural net model, 

python bag-of-words.py

Assignment\ 1/working/BagOfWords.hdf5 stores weights for the neural net
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

results folder contains the output of the code for the test images