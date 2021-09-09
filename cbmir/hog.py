import time
import pickle
import numpy as np
import cv2

from utils import find_img, print_progress
import dataset
from evaluation import *

from cbir_model import *

class HOGModel(CBIRModel):
    def __init__(self, hog_params=None, *args, **kwargs):
        super(HOGModel, self).__init__(*args, **kwargs)

        win_size = (256,256)
        block_size = (128, 128)
        block_stride = (64, 64)
        cell_size = (64, 64)
        n_bins = 15
        if hog_params is not None:
            self.hog_params = hog_params
        else:
            self.hog_params = win_size, block_size, block_stride, cell_size, n_bins
        self.hog = cv2.HOGDescriptor(*self.hog_params)

    def distance(self, a, B):
        """
        Return vector of distances (from vector a to each vector in matrix B).
        """
        return np.linalg.norm(a-B, axis=1, ord=1)

    def compute(self, img):
        """
        Compute HOG descriptor for given image.
        """
        hist = self.hog.compute(img)
        return hist.reshape(1, -1)

    def save_model(self, path):
        self.hog = None # can't pickle cv2.HOGDescriptor
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self.hog = cv2.HOGDescriptor(*self.hog_params)

    @staticmethod
    def load_model(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        model.hog = cv2.HOGDescriptor(*model.hog_params)
        return model

    def extract_dataset_descriptors(self):
        assert self.hog is not None

        start = time.time()
        des_all = []
        print("Extracting HOG descriptors...")
        for img_idx in range(self.n_imgs):
            if self.valid_idxs and img_idx in self.valid_idxs:
                des_all.append(np.zeros_like(des_all[-1]))
                continue # don't include validation images

            img = find_img(self.array_path, img_idx)
            hist = self.compute(img)
            des_all.append(hist)
            print_progress(img_idx, self.n_imgs)

        self.dataset = np.vstack(des_all).astype(np.float32)
        self.descriptor_size = self.dataset.shape[1]
        print("Saved HOG descriptors for the train dataset.")
        end = time.time()
        print("Time:", (end - start) / 60, "minutes")