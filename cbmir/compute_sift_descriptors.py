import time
import numpy as np
import cv2

from utils import *
import dataset

eps = 1e-8

def save_all_descriptors(n_imgs=dataset.n_imgs,
                         array_path=dataset.train_array_path,
                         valid_idxs=None, use_root_sift=True):
    if valid_idxs is not None:
        valid_idxs = set(valid_idxs)

    sift = cv2.SIFT_create()
    des_count = 0
    des_all = []

    start = time.time()
    print("Extracting SIFT descriptors...")
    for img_idx in range(n_imgs):
        if img_idx in valid_idxs:
            continue

        img = find_img(array_path, img_idx)
        kp, des = sift.detectAndCompute(img, None)
        if len(kp) == 0:
            continue

        if use_root_sift:
            des /= (des.sum(axis=1, keepdims=True) + eps)
            des = np.sqrt(des)

        des_count += des.shape[0]
        des_all.append(des)
        print_progress(img_idx, n_imgs)

    des_all = np.vstack(des_all)
    if use_root_sift:
        np.save("../features/train_rootsift_descriptors.npy", des_all)
    else:
        np.save("../features/train_sift_descriptors.npy", des_all)

    print("Done.")
    print("Saved", des_count, "descriptors for the train dataset.")
    end = time.time()
    print("Time:", (end - start) / 60, "minutes")

print("SIFT:")
save_all_descriptors(valid_idxs=dataset.valid_idxs, use_root_sift=False)

print("RootSIFT")
save_all_descriptors(valid_idxs=dataset.valid_idxs, use_root_sift=True)
