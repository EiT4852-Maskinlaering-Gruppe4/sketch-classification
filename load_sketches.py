import numpy as np
import os
from sklearn.utils import shuffle
import sklearn.model_selection

# Utility function to handle various os paths
def allround_path(path_to_data = ""):
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, path_to_data)
    return path


def load_sketches(train_ratio, path_to_data = "", doShuffle=True):
    
    path_to_data = allround_path(path_to_data)
    # Load sketches.
    cat_sketches = np.load(path_to_data + "cat.npy")
    dog_sketches = np.load(path_to_data + "dog.npy")
    flower_sketches = np.load(path_to_data + "flower.npy")
    house_sketches = np.load(path_to_data + "house.npy")
    sun_sketches = np.load(path_to_data + "sun.npy")
    
    
    # Calculating number of sketches in train and test sets for each category.    
    cat_no_train = int(cat_sketches.shape[0]*train_ratio)
    cat_no_test  = cat_sketches.shape[0] - cat_no_train
    
    dog_no_train = int(dog_sketches.shape[0]*train_ratio)
    dog_no_test  = dog_sketches.shape[0] - dog_no_train
    
    flower_no_train = int(flower_sketches.shape[0]*train_ratio)
    flower_no_test  = flower_sketches.shape[0] - flower_no_train
    
    house_no_train = int(house_sketches.shape[0]*train_ratio)
    house_no_test  = house_sketches.shape[0] - house_no_train
    
    sun_no_train = int(sun_sketches.shape[0]*train_ratio)
    sun_no_test  = sun_sketches.shape[0] - sun_no_train
    
    
    # Splitting sketches from each category into training and test set. 
    cat_train, cat_test = sklearn.model_selection.train_test_split(cat_sketches, test_size=cat_no_test,  
                                                                                  train_size=cat_no_train, random_state=None,
                                                                                  shuffle=True, stratify=None)
    
    
    dog_train, dog_test = sklearn.model_selection.train_test_split(dog_sketches, test_size=dog_no_test, 
                                                                                  train_size=dog_no_train, random_state=None,
                                                                                  shuffle=True, stratify=None)
    
    
    flower_train, flower_test = sklearn.model_selection.train_test_split(flower_sketches, test_size=flower_no_test, 
                                                                                  train_size=flower_no_train, random_state=None,
                                                                                  shuffle=True, stratify=None)
    
    
    house_train, house_test = sklearn.model_selection.train_test_split(house_sketches, test_size=house_no_test, 
                                                                                  train_size=house_no_train, random_state=None,
                                                                                  shuffle=True, stratify=None)
    
    
    sun_train, sun_test = sklearn.model_selection.train_test_split(sun_sketches, test_size=sun_no_test, 
                                                                                  train_size=sun_no_train, random_state=None,
                                                                                  shuffle=True, stratify=None)
    
    
    # Creating labels for training data.
    cat_train_label = np.array([0 for _ in range(len(cat_train))])
    dog_train_label = np.array([1 for _ in range(len(dog_train))])
    flower_train_label = np.array([2 for _ in range(len(flower_train))])
    house_train_label = np.array([3 for _ in range(len(house_train))])
    sun_train_label = np.array([4 for _ in range(len(sun_train))])
    
    
    train_labels = cat_train_label
    train_labels = np.append(train_labels, dog_train_label,0)
    train_labels = np.append(train_labels, flower_train_label,0)
    train_labels = np.append(train_labels, house_train_label,0)
    train_labels = np.append(train_labels, sun_train_label,0)
    
    
    # Creating labels for test data.
    cat_test_label = np.array([0 for _ in range(len(cat_test))])
    dog_test_label = np.array([1 for _ in range(len(dog_test))])
    flower_test_label = np.array([2 for _ in range(len(flower_test))])
    house_test_label = np.array([3 for _ in range(len(house_test))])
    sun_test_label = np.array([4 for _ in range(len(sun_test))])
    
    
    test_labels = cat_test_label
    test_labels = np.append(test_labels, dog_test_label,0)
    test_labels = np.append(test_labels, flower_test_label,0)
    test_labels = np.append(test_labels, house_test_label,0)
    test_labels = np.append(test_labels, sun_test_label,0)
    

    # Creating training and test sets.
    train_set = cat_train
    train_set = np.append(train_set, dog_train,0)
    train_set = np.append(train_set, flower_train,0)
    train_set = np.append(train_set, house_train,0)
    train_set = np.append(train_set, sun_train,0)
    
    test_set = cat_test
    test_set = np.append(test_set, dog_test,0)
    test_set = np.append(test_set, flower_test,0)
    test_set = np.append(test_set, house_test,0)
    test_set = np.append(test_set, sun_test,0)
    
    
    label_strings = ['cat','dog','flower','house','sun']
    
    if doShuffle:
        # Randomize order of data.
        train_set, train_labels = shuffle(train_set, train_labels)
        test_set, test_labels = shuffle(test_set, test_labels)
    
    return train_set, test_set, train_labels, test_labels, label_strings
    








