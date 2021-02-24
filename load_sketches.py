import numpy as np
from sklearn.utils import shuffle



def load_sketches(train_ratio, test_ratio, path_to_data = "", doShuffle=True):
    
    
    # Load sketches.
    cat_sketches = np.load(path_to_data + "cat.npy")
    dog_sketches = np.load(path_to_data + "dog.npy")
    flower_sketches = np.load(path_to_data + "flower.npy")
    house_sketches = np.load(path_to_data + "house.npy")
    sun_sketches = np.load(path_to_data + "sun.npy")
    
    
    # Calculating number of sketches in train, test and validation set for each category.    
    cat_no_train = int(cat_sketches.shape[0]*train_ratio)
    cat_no_test  = int(cat_sketches.shape[0]*test_ratio)
    cat_no_val   = cat_sketches.shape[0] - cat_no_train - cat_no_test
    
    dog_no_train = int(dog_sketches.shape[0]*train_ratio)
    dog_no_test  = int(dog_sketches.shape[0]*test_ratio)
    dog_no_val   = dog_sketches.shape[0] - dog_no_train - dog_no_test
    
    flower_no_train = int(flower_sketches.shape[0]*train_ratio)
    flower_no_test  = int(flower_sketches.shape[0]*test_ratio)
    flower_no_val   = flower_sketches.shape[0] - flower_no_train - flower_no_test
    
    house_no_train = int(house_sketches.shape[0]*train_ratio)
    house_no_test  = int(house_sketches.shape[0]*test_ratio)
    house_no_val   = house_sketches.shape[0] - house_no_train - house_no_test
    
    sun_no_train = int(sun_sketches.shape[0]*train_ratio)
    sun_no_test  = int(sun_sketches.shape[0]*test_ratio)
    sun_no_val   = sun_sketches.shape[0] - sun_no_train - sun_no_test
    
    
    # Splitting sketches from each category into training and test set. 
    
    cat_traintest, cat_val = sklearn.model_selection.train_test_split(cat_sketches,   test_size=cat_no_val, 
                                                                                      train_size = (cat_no_train + cat_no_test),
                                                                                      random_state=None,
                                                                                      shuffle=True, stratify=None)
    cat_train, cat_test = sklearn.model_selection.train_test_split(cat_traintest, test_size=cat_no_test, 
                                                                                  train_size=cat_no_train, random_state=None,
                                                                                  shuffle=True, stratify=None)
    
    
    
    dog_traintest, dog_val = sklearn.model_selection.train_test_split(dog_sketches,   test_size=dog_no_val, 
                                                                                      train_size = (dog_no_train + dog_no_test),
                                                                                      random_state=None,
                                                                                      shuffle=True, stratify=None)
    dog_train, dog_test = sklearn.model_selection.train_test_split(dog_traintest, test_size=dog_no_test, 
                                                                                  train_size=dog_no_train, random_state=None,
                                                                                  shuffle=True, stratify=None)
    
    
    flower_traintest, flower_val = sklearn.model_selection.train_test_split(flower_sketches,   test_size=flower_no_val, 
                                                                                      train_size = (flower_no_train + flower_no_test),
                                                                                      random_state=None,
                                                                                      shuffle=True, stratify=None)
    flower_train, flower_test = sklearn.model_selection.train_test_split(flower_traintest, test_size=flower_no_test, 
                                                                                  train_size=flower_no_train, random_state=None,
                                                                                  shuffle=True, stratify=None)
    
    
    house_traintest, house_val = sklearn.model_selection.train_test_split(house_sketches,   test_size=house_no_val, 
                                                                                      train_size = (house_no_train + house_no_test),
                                                                                      random_state=None,
                                                                                      shuffle=True, stratify=None)
    house_train, house_test = sklearn.model_selection.train_test_split(house_traintest, test_size=house_no_test, 
                                                                                  train_size=house_no_train, random_state=None,
                                                                                  shuffle=True, stratify=None)
    
    
    sun_traintest, sun_val = sklearn.model_selection.train_test_split(sun_sketches,   test_size=sun_no_val, 
                                                                                      train_size = (sun_no_train + sun_no_test),
                                                                                      random_state=None,
                                                                                      shuffle=True, stratify=None)
    sun_train, sun_test = sklearn.model_selection.train_test_split(sun_traintest, test_size=sun_no_test, 
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
    
    
    # Creating labels for validation data.
    cat_val_label = np.array([0 for _ in range(len(cat_val))])
    dog_val_label = np.array([1 for _ in range(len(dog_val))])
    flower_val_label = np.array([2 for _ in range(len(flower_val))])
    house_val_label = np.array([3 for _ in range(len(house_val))])
    sun_val_label = np.array([4 for _ in range(len(sun_val))])
    
    
    val_labels = cat_val_label
    val_labels = np.append(val_labels, dog_val_label,0)
    val_labels = np.append(val_labels, flower_val_label,0)
    val_labels = np.append(val_labels, house_val_label,0)
    val_labels = np.append(val_labels, sun_val_label,0)
    
    
    # Creating training, test and validation sets.
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
    
    val_set = cat_val
    val_set = np.append(val_set, dog_val,0)
    val_set = np.append(val_set, flower_val,0)
    val_set = np.append(val_set, house_val,0)
    val_set = np.append(val_set, sun_val,0)
    
    label_strings = ['cat','dog','flower','house','sun']
    
    if doShuffle:
        # Randomize order of data.
        train_set, train_labels = shuffle(train_set, train_labels)
        test_set, test_labels = shuffle(test_set, test_labels)
        val_set, val_labels = shuffle(val_set, val_labels)
    
    return train_set, test_set, val_set, train_labels, test_labels, val_labels, label_strings
    








