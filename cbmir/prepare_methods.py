import dataset
from bow import *
from hog import *
from cnn import *

"""
Extract dataset descriptors with BoVW, HOG and CNN.
(with the default algorithm parameters and dataset location)

Save the models.
"""

print("Bag-of-Visual-Words, n_words=4000")
bow = BoVWModel()
bow.build_vocabulary()
bow.extract_dataset_descriptors()
bow.save_model("./retrieval_methods/bow_final")
print()

print("HOG")
hog = HOGModel()
hog.extract_dataset_descriptors()
hog.save_model("./retrieval_methods/hog_final.pkl")
print()

print("CNN: SqueezeNet with average pooling (5x5)")
cnn = CNNModel()
cnn.extract_dataset_descriptors()
cnn.save_model("./retrieval_methods/cnn_final.pkl")