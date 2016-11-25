import os
import gzip
import pickle

import pandas as pd
import numpy as np

def get_data():
    train_path = "../data/training.csv"
    test_path = "../data/test.csv"

    ds = pd.read_csv(train_path)

    item = list(ds)
    ds = ds.as_matrix()
    data = ds[:,30]
    data_mat = np.zeros((data.shape[0], 9216))
    for i in range(data.shape[0]):
        data_mat[i,:] = np.array(list(map(int, data[i].split()))).astype('float32')
    data_mat = data_mat.astype('float32')
    label = ds[:,:30].astype('float32')
    label[np.isnan(label)] = 0
    mask = np.isnan(ds[:,:30].astype('float32')).astype('float32')

    return dict([("train", dict(data=data_mat, label=label, mask=mask))])
