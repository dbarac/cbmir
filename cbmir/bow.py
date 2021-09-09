import cv2
import numpy as np
import faiss
import pickle
import time
import datetime
from scipy.spatial.distance import cosine as cosine_distance
from scipy.spatial import distance

import dataset
from utils import *
from evaluation import *
from cbir_model import *

class BoVWModel(CBIRModel):
    """
    Bag-of-Visual-Words model with TF-IDF weighting.
    """

    def __init__(self, n_words=4000, use_root_sift=True, *args, **kwargs):
        super(BoVWModel, self).__init__(*args, **kwargs)
        self.eps = 1e-8
        self.n_words = n_words
        self.use_root_sift = use_root_sift
        self.index = None
        self.extractor = cv2.SIFT_create()
        self.word_occurences = None

    def distance(self, a, B):
        """
        Return vector of distances (from vector a to each vector in matrix B).

        Distance used: cosine
        """
        from scipy.spatial import distance
        return distance.cdist([a], B, "cosine")[0]

    def build_vocabulary(self, descriptors=None):
        """
        Extract local feature descriptors for train dataset images
        (if not provided).
        Build the visual vocabulary by running k-means on the extracted
        train dataset descriptors.
        """
        if descriptors is None:
            descriptors = []
            print("Extracting local features...")
            start = time.time()
            for img_idx in range(self.n_imgs):
                if self.valid_idxs and img_idx in self.valid_idxs:
                    continue # skip validation images

                img = find_img(dataset.train_array_path, img_idx)
                kps, img_des = self.extractor.detectAndCompute(img, None)
                if len(kps) == 0:
                    continue # no descriptors found

                if self.use_root_sift:
                    img_des /= (img_des.sum(axis=1, keepdims=True) + self.eps)
                    img_des = np.sqrt(img_des)

                descriptors.append(img_des)
                print_progress(img_idx, self.n_imgs)

            descriptors = np.vstack(descriptors)
            end = time.time()
            print("Time:", (end - start) / 60, "minutes")

        print("Local descriptor count.", descriptors.shape[0])
        print("Building visual vocabulary by running k-means...")
        print("Start time:", datetime.datetime.now())
        start = time.time()
        kmeans = faiss.Kmeans(d=descriptors.shape[1], k=self.n_words, niter=50, nredo=5)
        kmeans.train(descriptors)
        self.vocab = kmeans.centroids
        self.index = kmeans.index
        end = time.time()
        print("Done.")
        print("Time:", (end - start) / 60, "minutes")

    def find_visual_words(self, descriptors):
        """
        Quantize each descriptor to a visual word (cluster centroid)
        from the vocabulary.

        Return the index of the nearest visual word for each descriptor.
        """
        assert self.index is not None

        word_idxs = self.index.search(descriptors.astype(np.float32), 1)[1]
        return word_idxs.squeeze()

    def load_img(self, img_idx):
        return cv2.imread(self.image_dir + str(img_idx) + ".png")

    def extract_dataset_descriptors(self):
        """
        Calculate occurences for each visual words (for tf-idf weighting).
        Compute the BoVW descriptor for each image in the train dataset.
        """
        assert self.vocab is not None

        start = time.time()

        self.n_words_in_img = np.zeros(self.n_imgs)
        # n_word_occurences[i] is the number of images that contain i-th word 
        self.n_word_occurences = np.zeros(self.n_words)

        self.dataset = np.zeros((self.n_imgs, self.n_words), dtype=np.float32)

        print("Computing BoVW descriptors...")
        for img_idx in range(self.n_imgs):
            if self.valid_idxs is not None and img_idx in self.valid_idxs:
                continue # skip validation images
            img = find_img(dataset.train_array_path, img_idx)
            bow_descriptor = self.dataset[img_idx]

            kps, descriptors = self.extractor.detectAndCompute(img, None)

            if len(kps) == 0:
                continue # no keypoints found, leave the descriptor empty for this image

            if self.use_root_sift:
                descriptors /= (descriptors.sum(axis=1, keepdims=True) + self.eps)
                descriptors = np.sqrt(descriptors)

            self.n_words_in_img[img_idx] = len(descriptors)
            word_idxs = self.find_visual_words(descriptors)
            np.add.at(self.n_word_occurences, word_idxs, 1)
            np.add.at(bow_descriptor, word_idxs, 1)
            self.dataset[img_idx] = bow_descriptor

            print_progress(img_idx, self.n_imgs)

        if np.any(self.n_word_occurences == 0):
            eps = 1e-7
            self.n_word_occurences += eps

        # reweight histogram bins in in BoVW descriptor (TF-IDF)
        for img_idx in range(self.n_imgs):
            bow_descriptor = self.dataset[img_idx]
            n_words_in_img = self.n_words_in_img[img_idx]
            if n_words_in_img > 0:
                bow_descriptor *= np.log(self.n_imgs / self.n_word_occurences) / n_words_in_img
                self.dataset[img_idx] = bow_descriptor

        end = time.time()
        print("Training time:", (end - start) / 60, "minutes")
            
    def save_model(self, path):
        extractor = self.extractor
        self.extractor = None # can't pickle cv2.SIFT

        faiss.write_index(self.index, path + ".idx")
        index = self.index
        self.index = None

        with open(path + ".pkl", "wb") as f:
            pickle.dump(self, f)

        self.extractor = extractor
        self.index = index

    @staticmethod
    def load_model(path):
        """
        The path should not have a file extension. (.idx and .pkl are added when loading)
        """
        with open(path + ".pkl", "rb") as f:
            model = pickle.load(f)
        model.extractor = cv2.SIFT_create()
        model.index = faiss.read_index(path + ".idx")
        return model

    def compute(self, img):
        """
        Compute BoVW descriptor for a single query image.
        """
        kps, descriptors = self.extractor.detectAndCompute(img, None)

        bow_descriptor = np.zeros(self.n_words)
        n_words_in_img = len(kps)
        if n_words_in_img == 0:
            return bow_descriptor

        if self.use_root_sift:
            descriptors /= (descriptors.sum(axis=1, keepdims=True) + self.eps)
            descriptors = np.sqrt(descriptors)
        word_idxs = self.find_visual_words(descriptors)
        np.add.at(bow_descriptor, word_idxs, 1)
        
        # reweight (TF-IDF)
        bow_descriptor *= np.log(self.n_imgs / self.n_word_occurences) / n_words_in_img
        return bow_descriptor