import numpy as np
import os
from sklearn.utils import shuffle

# Utility function to handle various os paths
def allroundPath(path_to_data = ""):
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, path_to_data)
    return path

def load_sketches(path_to_data = "", doShuffle=True):
    
    path_to_data = allroundPath(path_to_data)

    cat = np.load(path_to_data + "cat.npy")
    dog = np.load(path_to_data + "dog.npy")
    flower = np.load(path_to_data + "flower.npy")
    house = np.load(path_to_data + "house.npy")
    sun = np.load(path_to_data + "sun.npy")


    print("Number of cat sketches: ", len(cat))
    print("Number of dog sketches: ", len(dog))
    print("Number of flower sketches: ", len(flower))
    print("Number of house sketches: ", len(house))
    print("Number of sun sketches: ", len(sun))

    sketches = cat
    sketches = np.append(sketches, dog,0)
    sketches = np.append(sketches, flower,0)
    sketches = np.append(sketches, house,0)
    sketches = np.append(sketches, sun,0)

    print("Total number of sketches: ", len(sketches))

    label_strings = ['cat','dog','flower','house','sun']

    cat_label = np.array([0 for _ in range(len(cat))])
    dog_label = np.array([1 for _ in range(len(dog))])
    flower_label = np.array([2 for _ in range(len(flower))])
    house_label = np.array([3 for _ in range(len(house))])
    sun_label = np.array([4 for _ in range(len(sun))])

    sketch_labels = cat_label
    sketch_labels = np.append(sketch_labels, dog_label,0)
    sketch_labels = np.append(sketch_labels, flower_label,0)
    sketch_labels = np.append(sketch_labels, house_label,0)
    sketch_labels = np.append(sketch_labels, sun_label,0)

    print("Total number of labels: ", len(sketch_labels))
    if doShuffle:
        sketches, sketch_labels = shuffle(sketches, sketch_labels)

    return sketches, sketch_labels, label_strings
