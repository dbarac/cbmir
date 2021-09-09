import dataset
from utils import *
from cnn import *
from hog import *

# retrieval methods
cnn = CNNModel.load_model("./retrieval_methods/cnn_final.pkl")
hog = HOGModel.load_model("./retrieval_methods/hog_final.pkl")

path = "../data/test_images/585.png"
query_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

k = 5
retrieved_idxs = cnn.find_knn(query_img, k)
reranked_idxs = hog.sort_by_distance(query_img, retrieved_idxs)

imgs = get_neighbour_imgs(
    query_img, dataset.train_array_path, retrieved_idxs, k
)
imgs = make_image_grid(imgs, n_per_row=k+1)
cv2.imwrite("result.png", np.transpose(imgs, (1, 2, 0)))
