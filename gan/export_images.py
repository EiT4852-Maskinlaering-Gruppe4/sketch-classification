import os
import cv2
import numpy as np

category_name = "Dog"
goal_directory = f"{os.getcwd()}/gan/Images/{category_name}"
extracted_directory = "PATH/TO/UNZIPPED/FOLDER"

def extract_images():

    

    # Create goal folder and move images to correct path
    if not os.path.exists(goal_directory):
        os.makedirs(goal_directory)


    for root, dirs, files in os.walk(extracted_directory):
        for file in files:
            if not file.endswith("tar"):
                old_path = os.path.join(root, file)
                print(old_path)
                # print(old_path)
                new_path = os.path.join(goal_directory, file)
                os.replace(old_path, new_path)


def import_images(directory, img_size, n_images):

    RESIZE_TO = img_size
    img_array = []

    print("Importing images from ", directory)
    print("Image size: ", RESIZE_TO)

    counter = 0
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        img_array.append(cv2.resize(cv2.imread(path), RESIZE_TO))
        
        counter += 1
        if counter >= n_images:
            break

    img_array = np.array(img_array)
    # print(img_array[0])
    # cv2.imshow("hei", img_array[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img_array


def main():
    extract_images()