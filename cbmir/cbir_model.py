import time
import pickle
import numpy as np
import cv2

from utils import find_img
import dataset

class CBIRModel():
    """
    Image retrieval base class.
    """
    def __init__(self, n_imgs=dataset.n_imgs, image_dir=dataset.image_dir,
                 array_path=dataset.train_array_path, valid_idxs=None):

        self.n_imgs = n_imgs
        self.array_path = array_path
        self.image_dir = image_dir
        self.dataset = None

        # save indices of images from the validation dataset
        if valid_idxs is not None:
            self.valid_idxs = set(valid_idxs)
        else:
            self.valid_idxs = None

    def distance(self, a, B):
        """
        Return distances (from vector a to each vector in matrix B).
        """
        pass

    def compute(self, img):
        """
        Compute the descriptor for given image.
        """
        pass

    def save_model(self, path):
        pass

    @staticmethod
    def load_model(path):
        pass

    def find_knn(self, img, k):
        query_desc = self.compute(img)

        dists = self.distance(query_desc, self.dataset)
        sorted_idxs = np.argsort(dists)
        return list(sorted_idxs[:k])

    def sort_by_distance(self, img, idxs):
        """
        Compute the descriptor for img.
        For each image index in idxs, compute the distance between the
        descriptor stored for that image and the query image descriptor.
        Finally, sort the images by distance and return their indices.

        This can be useful for re-ranking images retrieved with another method.
        """
        assert type(idxs) == list
        query_desc = self.compute(img)

        descriptors = self.dataset[idxs]
        dists = self.distance(query_desc, descriptors)

        idxs = np.array(idxs)
        sorted_idxs = idxs[np.argsort(dists)]
        return list(sorted_idxs)