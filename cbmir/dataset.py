import pandas as pd
import numpy as np

image_dir = "../data/images/"
test_image_dir = "../data/test_images/"

n_imgs = 30000
n_test_imgs = 8000

train_df = pd.read_pickle("../data/df_paper_fixed_stage_2")
test_df = pd.read_pickle("../data/df_paper_fixed_test_stage_2")

train_array_path = "../data/images_matrix_single.npy"
test_array_path = "../data/test_images_matrix_single.npy"

# A part of the training dataset is used for validation.
# (models are trained only on train_idxs)
# For the final testing the models are trained with the whole train dataset.
# (both train_idxs and valid_ixds)
valid_idxs = np.loadtxt("../data/valid_indices.txt").astype(int)
train_idxs = np.loadtxt("../data/train_indices.txt").astype(int)

test_idxs = np.arange(n_test_imgs).astype(int)
