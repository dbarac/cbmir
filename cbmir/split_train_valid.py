import numpy as np
import pandas as pd

import dataset

df = dataset.train_df

def split_train_valid(df, valid_size=0.2):
    """
    Split the dataset for training and validation.
    The validation dataset will contain (valid_size*100)%
    of instances of each class (randomly selected).
    
    Returns: 
     - (train_idxs, valid_idxs): a tuple of lists
       which contain indices (not IDs) belonging to the dataset.
       The indices can be used to select images from the raw data array
       or to select dataframe rows with df.iloc[idxs].
    """
    np.random.seed(123)
    
    valid_idxs = []
    train_idxs = []
    for body_part in df["BodyPartExaminedFixed"].unique():
        rows = df.loc[(df["BodyPartExaminedFixed"] == body_part)]
        idxs = np.in1d(df.index, rows.index).nonzero()[0]
        idxs = np.unique(idxs)
        
        valid_len = int(len(idxs) * valid_size)
        print(body_part, valid_len)
        idxs = np.random.permutation(idxs)
        valid_idxs += list(idxs[:valid_len])
        train_idxs += list(idxs[valid_len:])
    
    print(len(valid_idxs) + len(train_idxs))
    return train_idxs, valid_idxs
 

def save(train_idxs, valid_idxs):
    train_idxs = np.array(train_idxs)
    valid_idxs = np.array(valid_idxs)

    print(train_idxs.dtype)
    np.savetxt("../data/train_indices.txt", train_idxs, fmt="%ld")
    np.savetxt("../data/valid_indices.txt", valid_idxs)


def main():
    train_idxs, valid_idxs = split_train_valid(df)

    print(train_idxs[:3], valid_idxs[:3])
    save(train_idxs, valid_idxs)


if __name__ == "__main__":
    main()
