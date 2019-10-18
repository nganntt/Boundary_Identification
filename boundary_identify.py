from sklearn.model_selection import train_test_split
from clustering import cluster_scenarios
from scenario import gen_testcase
from random import randint
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
#from association_rule import decode_testcase, convert_dataFrame, apriori_speed, rules_frequenItemsets
from association_rule import apriori_speed_rs, rules_frequenItemsets_rs, decode_testcase_in_boundary_list
from sys_out import read_samples_from_queries, convert_boundary_dataframe_normalize, convert_boundary_dataframe_beamNG
from activeLearning_GPC import pool_base_sampling
from clustering import cluster_scenarios, divide_subgroups

EPSILON = 0.07


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
                boundarys_pair = (*item, *X_train[boundary_distance_idx[i],:])
                #print ("X train for nearest neighbor",X_train[boundary_distance_idx[i],:])
                boundarys.append(boundarys_pair)
    return boundarys


def find_boundary_elements(labled_queries):
    '''
    Find the boundary from sampled which  which are collected in AL process
    :param labled_queries: samples are labeled safe or unsafe
    :return: a list of elements which located on the boundary
    '''
    mask_safe = (labled_queries[:, 4] == 1)
    mask_unsafe = (labled_queries[:, 4] == 0)
    safe_testcases = labled_queries[mask_safe]
    unsafe_testcases = labled_queries[mask_unsafe]

    # cluster safe and unsafe testcases into subgroups
    X_safe_testcases = safe_testcases[:, :4]
    X_unsafe_testcases = unsafe_testcases[:, :4]
    sub_groups_safe, num_sub_group_safe = cluster_scenarios(X_safe_testcases)
    sub_groups_unsafe, num_sub_group_unsafe = cluster_scenarios(X_unsafe_testcases)

    subgroups_in_list_safe = divide_subgroups(sub_groups_safe, num_sub_group_safe)
    subgroups_in_list_unsafe = divide_subgroups(sub_groups_unsafe, num_sub_group_unsafe)

    boundary_list = list()
    for i in range(num_sub_group_safe):
        for j in range(num_sub_group_unsafe):
            print("\n\n boundary subgroups %s safe and %s unsafe \n" % (i, j))
            boundary_sub = find_boundary_bw_subclusters(subgroups_in_list_safe[i],
                                                        subgroups_in_list_unsafe[j])
            boundary_list = boundary_list + boundary_sub

    return boundary_list


def find_boundary_elements_twoWays(*args, online = None, sampling_list = None):
    '''
    identify the boundary from the samples which are collected in AL process
    by online method or offiline (online = run sampling process, offline = get samples which
    were sampled and save in the log file)
    :param sampling_list:
    :return: a list of pair testcases (safe and unsafe) which belongs the boundary
    '''
    params_sampling = [arg for arg in args]
    print (params_sampling)
    boundary_list_safe_unsafe = list()
    #finding the list of boundary elements
    if online ==None:
        #get testcases from sampling process with active learning
        labled_queries = read_samples_from_queries('Uncertainty_queries_10_17_2019__024059.cvs')
        boundary_list_safe_unsafe = find_boundary_elements(labled_queries)
        #print("\n boundary list \n", boundary_list_safe_unsafe)
        #print("\n boundary list shape \n", boundary_list_safe_unsafe.__len__())

    else:
        #set parameter like this
        # pool_base_sampling(1000, 3, 500, 'left_curve', 'normalize_left_curve.cvs', 'uncertainty')
        sample_uncer, performance_uncer = pool_base_sampling(int(params_sampling[0]),
                                                             int(params_sampling[1]), int(params_sampling[2]),
                                                             params_sampling[3],
                                                             params_sampling[4], params_sampling[5])
        boundary_list_safe_unsafe = find_boundary_elements(sample_uncer)

    return boundary_list_safe_unsafe



def main(*args, online=None):
    boundary = list()
    if online ==None:
        boundary = find_boundary_elements_twoWays(online=None)
        print_boundary(boundary)

    else:
        boundary = find_boundary_elements(*args, online='yes')
        print_boundary(boundary)

    convert_boundary_dataframe_normalize(boundary, "Boundary_testcases_norminalize_")
    convert_boundary_dataframe_beamNG(boundary, "Boundary_testcases_beamnNG_")

    # tc_encode_association_rule, testcaes_beamng = decode_testcase_in_boundary_list(boundary)
    # frequent_items = apriori_speed_rs(tc_encode_association_rule)
    # rules_frequenItemsets_rs(frequent_items)





if __name__ == "__main__":
    main(sys.argv[1])


#find the boundary by running active learning to get samples data, then identify the boundary based on
#found data. Command to run
# The parameter in the command is the parameters to call pool_base_sampling function in activeLearning_GPC)
# python boundary_identification 1000, 3, 500, 'left_curve', 'normalize_left_curve.cvs', 'uncertainty', online = 'yes'
