from sklearn.model_selection import train_test_split
from clustering import cluster_scenarios
from scenario import gen_testcase
from random import randint
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
#from association_rule import decode_testcase, convert_dataFrame, apriori_speed, rules_frequenItemsets
from AR_speed_safe import decode_testcase, apriori_speed_rs, rules_frequenItemsets_rs

EPSILON = 0.25


def create_tmp_list_tc_testing(num_sampels, shape):
    num_pool = num_sampels
    road_shape = shape
    beanNG_tc_list, normalize_tc_list  = gen_testcase(num_pool, road_shape)
    result = [randint(0,1) for i in range(num_pool)]
    list_tc = cluster_scenarios(normalize_tc_list, result)
    print("\n Subgroup of testcases: ", list_tc)
    return list_tc


def print_boundary(bounary_list):
    for i,item in enumerate(bounary_list,1):
        print("\n %s, %s"%(i, str(item)))


def check_boundary_region(distance_item, EPSILON):
    '''
    Verifying 2 testcases belongs the boundary
    :param distance_item: distance between two testcases
    :param EPSILON: threshold to check whether 2 testcases are close enough to put them into the boundary group or not
    :return:
    '''
    mask_distance = distance_item[0] <= EPSILON
    boundary_distance_idx = distance_item[1][mask_distance]
    return boundary_distance_idx


def find_boundary_bw_subclusters(labeled_testcase1, labeled_testcase2):
    '''
    Find the boundary betweens safe and unsafe subgroups
    add the testcases which their distances < epsilon_threshold to the boundary list
    :param labeled_testcase1: a testcase in safe group testcase
    :param labeled_testcase2: a testcase in unsafe group testcase
    :return: a list of pair testcases (safe and unsafe testcase)
    '''
    X_train = labeled_testcase1[:,:4]
    y_train = labeled_testcase1[:,-1]
    #Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)

    #Train the model using the training sets
    knn.fit(X_train, y_train)

    #get distance and k nearest neighbors: a tuple
    #(array([[0.        , 0.47235039, 0.47500414]]), array([[0, 2, 7]], dtype=int64))
    #distance = list()
    X_labeled_testcase2 = labeled_testcase2[:,:4]
    boundarys = list()
    for item in X_labeled_testcase2:
        distance_item = knn.kneighbors([item]) 
        print("item", item)
        print ("distance item", distance_item)
        #distance.append(distance_item)
        boundary_distance_idx = check_boundary_region(distance_item, EPSILON)
        #print ("distanc idx", boundary_distance_idx)
        if (len(boundary_distance_idx) != 0 ):
            for i in range(len(boundary_distance_idx)):
                boundarys_pair = (item, X_train[boundary_distance_idx[i],:])
                #print ("X train for nearest neighbor",X_train[boundary_distance_idx[i],:])
                boundarys.append(boundarys_pair)
    return boundarys


def find_boundary(sampling_list):
    '''
    identify the boundary from the samples which are collected in AL process
    :param sampling_list:
    :return: a list of pair testcases which belongs the boundary
    '''
    pass

list_tc_1a = create_tmp_list_tc_testing(100, 'right_curve')
list_tc_1b = create_tmp_list_tc_testing(100, 'right_curve')
boundary1 = find_boundary_bw_subclusters(list_tc_1a, list_tc_1b)
#boundary2 = find_boundary_bw_subclusters(list_tc_2a, list_tc_2b)
#boundary3 = find_boundary_bw_subclusters(list_tc_3a, list_tc_3b)
#boundary4 = find_boundary_bw_subclusters(list_tc_4a, list_tc_4b)

boundary = boundary1 #+ boundary2 + boundary3 + boundary4

#print_boundary(boundary)

dtc = decode_testcase(boundary)
print("\n\n\n The boundary list:  \n")
#print_boundary(dtc)

frequent_items = apriori_speed_rs(dtc)
rules_frequenItemsets_rs(frequent_items)




