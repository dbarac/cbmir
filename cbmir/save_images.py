import time
import numpy as np
import cv2

from utils import *
import dataset

print("Loading train images...")
train_imgs = np.load(dataset.train_array_path)
print("Done.")

for i, img in enumerate(train_imgs):
    img = np.transpose(img, (1,2,0))
    img = (img * 255).astype('uint8')
    print_progress(i, dataset.n_imgs)
    path = "../data/images/" + str(i) + ".png"
    cv2.imwrite(path, img)

del train_imgs

print("Loading test images...")
test_imgs = np.load(dataset.test_array_path)
print("Done.")

for i, img in enumerate(test_imgs):
    img = np.transpose(img, (1,2,0))
    img = (img * 255).astype('uint8')
    print_progress(i, dataset.n_test_imgs)
    path = "../data/test_images/" + str(i) + ".png"
    cv2.imwrite(path, img)