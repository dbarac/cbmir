import numpy as np
import pickle

import dataset
from evaluation import *
from bow import *
from cnn import *
from hog import *


def evaluate_retrieval(label, param_str, model, k, rerank_model=None, k_prep=None):
    """
    Retrieve similar images with the given model. Evaluate with mP and mAP on the test dataset.
    Save the retrieved image indices.
    """
    print(label)

    retrieved_knns = []
    start = time.time()
    print("Finding nearest neighbours...")
    for i, img_idx in enumerate(dataset.test_idxs):
        img = find_img(dataset.test_array_path, img_idx)

        if rerank_model and k_prep:
            retreived = model.find_knn(img, k_prep)
            reranked = rerank_model.sort_by_distance(img, retreived)[:k]
            retrieved_knns.append(reranked)
        else:
            retrieved_knns.append(model.find_knn(img, k))
        print_progress(i, len(dataset.test_idxs))

    retrieved_path = "./retrieval_methods/" + param_str + "_retrieved.npy"
    np.save(retrieved_path, np.array(retrieved_knns))

    print("Evaluating...")
    results = evaluate(
        dataset.train_df, dataset.test_df,
        dataset.train_idxs, dataset.test_idxs,
        retrieved_knns, max_k=k
    )
    print("Done.")

    end = time.time()
    print("Evaluation time:", (end - start) / 60, "minutes")
    print()

    return results

bow = BoVWModel.load_model("./retrieval_methods/bow_final")
hog = HOGModel.load_model("./retrieval_methods/hog_final.pkl")
cnn = CNNModel.load_model("./retrieval_methods/cnn_final.pkl")

k = 10
model_results = []
model_labels = []

# BoVW
label = "Bag-of-Visual-Words"
param_str = "bow"
results = evaluate_retrieval(label, param_str, bow, k)
model_labels.append(label)
model_results.append(results)

# HOG
label = "HOG"
param_str = "hog"
results = evaluate_retrieval(label, param_str, hog, k)
model_labels.append(label)
model_results.append(results)


# CNN
label = "CNN (SqueezeNet + average pool 5x5)"
param_str = "cnn"
results = evaluate_retrieval(label, param_str, cnn, k)
model_labels.append(label)
model_results.append(results)

# Best found re-ranking combination (retrieval model, rerank model, k_prep)
label = "Retrieved with CNN, re-ranked with HOG, k_prep=10)"
param_str = "rerank"
results = evaluate_retrieval(label, param_str, cnn, k, hog, k_prep=10)
model_labels.append(label)
model_results.append(results)

BODY_PART = 0
MODALITY = 1
BOTH = 2
model_results = np.array(model_results)

results_path = "../eval_results/final_test_results.pkl"
with open(results_path, "wb") as f:
    pickle.dump((model_labels, model_results), f)

title = "Retrieval performance on the test dataset"
save_path = "../eval_results/final_comparison"
plot_model_comparison(model_labels, model_results[:, BOTH, :], title, k, save_path, y_lim=[0.5, 1])
