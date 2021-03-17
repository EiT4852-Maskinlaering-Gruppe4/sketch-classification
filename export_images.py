import os
import cv2
import numpy as np


category_name = "Dog"
goal_directory = f"{os.getcwd()}/GAN/Images/{category_name}"
extracted_directory = "C:/Users/kri_k/OneDrive/Documents/Images/Images"


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


RESIZE_TO = 100, 100
img_array = []

for file in os.listdir(goal_directory):
    path = os.path.join(goal_directory, file)
    img_array.append(cv2.resize(cv2.imread(path), RESIZE_TO))
    break

img_array = np.array(img_array)
print(img_array[0])
cv2.imshow("hei", img_array[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
