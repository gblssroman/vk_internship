import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

def prepare_data():
    print("\nAssuming you have test.csv file in the right format...")
    test = pd.read_csv('datasets/test.csv')
    features = pd.read_csv('datasets/features.csv')
    with open('cols_to_drop.pkl', 'rb') as high_corr_cols:
        cols_to_drop = pickle.load(high_corr_cols)

    knn = KNeighborsClassifier(n_neighbors=2)
    Xknn = features[['lat', 'lon']]
    knn.fit(Xknn, np.arange(features.shape[0]))
    nearest = knn.predict(test.iloc[:, 1:3])
    test['bound'] = nearest
    features['bound'] = np.arange(features.shape[0])
    data_to_pred = pd.merge(test, features, how='left', on='bound').\
        drop(columns=['bound', 'lat_y', 'lon_y']).\
        drop(columns=cols_to_drop)
    return data_to_pred



