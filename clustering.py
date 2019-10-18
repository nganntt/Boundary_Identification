print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scenario import gen_testcase
from random import randint
import matplotlib.pyplot as plt

def cluster_scenarios(X):
    '''
    Cluster testcases in safe or unsafe group into subgroups
    :param normalize_tc_list:  feature of the testcases (X)
    :param result_tc: result of testcase
    :return: subgroups of testcases which have similar features
    '''

    # #############################################################################

    X_tranform = StandardScaler().fit_transform(X) # data already normalize

    # #############################################################################
    # Compute DBSCAN
    #perfect number of eps = 0,7, min_sample = 3
    db = DBSCAN(eps=0.7, min_samples=3).fit(X_tranform)

    #print ("database print....", db)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
   

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)  #count number of noice
    clusters = [X[labels == i] for i in range(n_clusters_)]
    
    clusters_dbResults = list()
    for i in range(n_clusters_):
        tmp = np.full((clusters[i].shape[0], clusters[i].shape[1]+1),float(i))
        tmp[:,:-1] = clusters[i]
        clusters_dbResults.append(tmp)
        
    clusters_dbResults = np.concatenate(clusters_dbResults)
    num_group = clusters_dbResults.shape[1]
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.5f"
          % metrics.silhouette_score(X, labels))
    return clusters_dbResults, num_group

def divide_subgroups(clusters, num_subgroup):
    sub_groups = list()
    for i in range(num_subgroup):
        mask_subgroup = (clusters[:,4] ==i)
        subgroup = clusters[mask_subgroup]
        sub_groups.append(subgroup)
    return sub_groups