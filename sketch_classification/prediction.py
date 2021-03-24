from keras.models import load_model

from load_sketches import load_sketches
import numpy as np
import matplotlib.pyplot as plt
import cv2

CATEGORIES = ['cat', 'dog', 'flower', 'house', 'sun']
# Var mye rot med å bruke relativ path, derfor absolute path til modell
#PATH_TO_MODEL = '/Users/ralfleistad/Desktop/Skole/EiT/prosjekt/sketch-classification/sketch_classification/models/ralf/b32-e9/b32-e9.h5'
# For testing on test_set from trainingdata
PATH_TO_DATA = 'sketches/'

PATH_TO_MODEL = '/Users/ralfleistad/Desktop/Skole/EiT/prosjekt/sketch-classification/sketch_classification/models/ralf/TEMP.h5'

def pre_process():
    img_path = '/Users/ralfleistad/Downloads/test_imgs/dog.png'

    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(28, 28))

    #plt.imshow(img)
    #plt.show()
    img = np.reshape(img, (-1, 28, 28, 1))
    return img


PATH_TO_DATA = 'sketches/'

train_set, test_set, train_labels, test_labels, label_strings = load_sketches(0.7, PATH_TO_DATA)
train_set = np.reshape(train_set, (train_set.shape[0], 28, 28, 1))
test_set = np.reshape(test_set, (test_set.shape[0], 28, 28, 1))

model = load_model(PATH_TO_MODEL)

#pred = model.predict_classes(test_set)
#imgplot = plt.imshow(test_set[0])
#plt.show()
#print(pred[0])

pred = model.predict_classes(pre_process())
print(CATEGORIES[pred[0]])