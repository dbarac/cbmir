import torch
import torchvision
from torchvision import models
import time
import torch.nn as nn

import dataset
from utils import *
from cbir_model import *
from evaluation import *

torch.set_grad_enabled(False)

class CNNModel(CBIRModel):
    """
    CNN image retrieval.
    Use a convolutional neural network to extract image descriptors.
    """
    def __init__(self, model=None, avgpool_kernel_size=5, *args, **kwargs):
        super(CNNModel, self).__init__(*args, **kwargs)
        
        if model is None:
            model = models.squeezenet1_0(pretrained=True)

        # drop the fully-connected layers at the end of the net
        cnn_layers = list(model.children())
        extractor_layers = cnn_layers[:-1]

        # optionally, reduce image descriptor size by adding
        # an average pooling layer at the end
        if avgpool_kernel_size is not None:
            self.extractor = nn.Sequential(
                *extractor_layers,
                nn.AvgPool2d(avgpool_kernel_size)
            )
        else:
            self.extractor = nn.Sequential(*extractor_layers)

    def distance(self, a, B):
        """
        Return vector of distances (from vector a to each vector in matrix B).
        """
        return np.linalg.norm(a-B, axis=1, ord=2)

    def compute(self, img):
        """
        Extract a descriptor with a pretrained CNN.
        """
        img = img.astype(np.float32) / 255
        img = np.stack((img, img, img)) # pretrained convnets require 3 channel imgs
        img = img[np.newaxis, ...]
        img = torch.from_numpy(img)
        descriptor = self.extractor(img).flatten().numpy()
        return descriptor

    def extract_dataset_descriptors(self):
        start = time.time()
        des_all = []
        print("Extracting CNN descriptors...")
        for img_idx in range(self.n_imgs):
            if self.valid_idxs and img_idx in self.valid_idxs:
                des_all.append(np.zeros_like(des_all[-1]))
                continue # don't include validation images

            img = find_img(self.array_path, img_idx)
            des = self.compute(img)
            des_all.append(des)
            print_progress(img_idx, self.n_imgs)

        self.dataset = np.vstack(des_all).astype(np.float32)
        self.descriptor_size = self.dataset.shape[1]
        print("Saved CNN descriptors for the train dataset.")
        end = time.time()
        print("Time:", (end - start) / 60, "minutes")

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
