## Retrieval methods
* `cbir_model.py`: image retrieval base class
* `bow.py`: Bag-of-Visual-Words (BoVW) implementation
* `hog.py`: HOG (Histogram-of-oriented-gradients) retrieval implementation
* `cnn.py`: retrieval with CNN-extracted descriptors

## Scripts
* `prepare_methods.py`: build visual vocabulary for BoVW, extract descriptors with all retrieval methods
   and save for later use
* `save_images.py`: save images from the train and test raw data arrays
* `test.py`: evaluate the retrieval methods on the test dataset
* `compute_sift_descriptors.py`: extract SIFT and RootSIFT descriptors for later reuse
* `split_train_valid.py`: select random images for validation, save train and validation indices
* `example.py`: short image retrieval example

## Other
* `image-retrieval.ipynb`: notebook with an image retrieval example
* `dataset.py`: dataset default path definitions
* `evaluation.py`: implementation of evaluation metrics
* `utils.py`: loading images, make image grid, print progress
