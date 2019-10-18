
import os
from datetime import datetime
import pandas as pd
import numpy as np
from association_rule import decode_testcase_in_boundary_list

def print_star_end(msg):
    print("======================================================================================================= ")
    print("******************************************************************************************************* ")
    print ("\n    %s \n" %msg)
    print("******************************************************************************************************* ")
    print("======================================================================================================= ")
        

def print_title(msg):
    print("=======================================================================================================")
    print ("       " +msg)
    print("======================================================================================================= ")
        
def print_sub_tit(msg):
    print("--------------------------------------------------------------- ")
    print ("       " +msg)
    print("--------------------------------------------------------------- ")
        
def trace_func(file_name, function):
    print ("\nFile name: " + file_name + ", funciton: "+ function)
    
    
#function only for print a list
def print_collection(data,len):
    for i in range(len):
        print ("     "+ str(data[i]))
        
        
def warning (msg):
    print("\n ******************************************************************")
    print("WARNING:   " , msg)
    print("\n ******************************************************************")
    
    
def result (msg):
    print("======================================================================================================= ")
    print("*******************************************************************************************************")
    print("\n VERDICT: %s \n" % msg)
    print("*******************************************************************************************************")
    print("======================================================================================================= ")
    
def write_data_queries_file(queries, file_name):
    '''
    Write queries to cvs file
    :param queries:
    :param file_name:
    :return:
    '''
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y__%H%M%S")
    path_dir = os.getcwd()
    file_name_cvs = os.path.join(path_dir, "store", file_name + '_' + date_time + ".cvs")


    normalize_tc = pd.DataFrame(queries)
    normalize_tc = normalize_tc.set_axis(['Road_shape', 'speed', 'light',
                                          'weather', 'result'], axis=1, inplace=False)
    normalize_tc.to_csv(file_name_cvs)

def write_perfromance(perf_list, file_name):
    '''
    Write performance to svs file
    :param perf_list:
    :param file_name:
    :return:
    '''
    performance = np.asarray(perf_list)
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y__%H%M%S")
    path_dir = os.getcwd()
    file_name_cvs = os.path.join(path_dir, "store", file_name + '_' + date_time + ".cvs")
    df = pd.DataFrame(performance)
    df = df.set_axis(['NonError', 'Error', 'Accuracy'], axis=1, inplace=False)
    df.to_csv(file_name_cvs)

def read_samples_from_queries(file):
    '''
       Read data of the performance after the sampling process
       Data is saved in cvs file
       '''
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    store_dir = os.path.join(curr_dir, 'store')
    file_name = file_name = os.path.join(store_dir, file)
    df = pd.read_csv(file_name)
    feature_col = ['Road_shape', 'speed', 'light', 'weather', 'result']
    queries = df.loc[:, feature_col]
    return queries.to_numpy()

def convert_boundary_dataframe_normalize(boundary_list_safe_unsafe,file_name):
    boundary = np.asarray(boundary_list_safe_unsafe)
    num_boundary_pair = int(boundary.shape[0])
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y__%H%M%S")
    path_dir = os.getcwd()
    file_name_cvs = os.path.join(path_dir, "store", file_name + '_' + date_time + ".cvs")

    safe_list_tc = boundary[:,0:4]
    unsafe_list_tc = boundary[:, 4:]
    boundary_combine = np.zeros((num_boundary_pair, 10))
    boundary_combine[:,0:4] = safe_list_tc
    boundary_combine[:, 4] = 1
    boundary_combine[:,5:9] = unsafe_list_tc
    boundary_combine[:, -1] = 0

    normalize_tc = pd.DataFrame(boundary_combine)
    normalize_tc = normalize_tc.set_axis(['Road_shape1', 'speed1', 'light1',  'weather1',
                                          'safe_state',
                                          'Road_shape2', 'speed2', 'light2', 'weather2',
                                          'unsafe_state', ], axis=1, inplace=False)
    normalize_tc.to_csv(file_name_cvs)




def convert_boundary_dataframe_beamNG(boundary_list_safe_unsafe,file_name):
    _, beamNG_list_tc = decode_testcase_in_boundary_list(boundary_list_safe_unsafe)
    print("\n \n \n list beamNG", beamNG_list_tc)
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y__%H%M%S")
    path_dir = os.getcwd()
    file_name_cvs = os.path.join(path_dir, "store", file_name + '_' + date_time + ".cvs")

    beamNG_list_tc = np.asarray(beamNG_list_tc)

    beamNG_tc_df = pd.DataFrame(beamNG_list_tc)
    beamNG_tc_df = beamNG_tc_df.set_axis(['Road_shape1', 'speed1', 'light1',  'weather1',
                                          'safe_state',
                                          'Road_shape2', 'speed2', 'light2', 'weather2',
                                          'unsafe_state', ], axis=1, inplace=False)
    beamNG_tc_df.to_csv(file_name_cvs)
    # print("\n item1 ", safe_list_tc[0,:])
    # print("\n item1", unsafe_list_tc[0,:])
    # print("\n\n combine  \n", boundary_combine[0,:])




def summary (list_tc):
    print_star_end("                 SUMMARY     ")
    msg = ""
    for idx, tc in enumerate(list_tc):
        if tc[4]:
            msg = "FAILED"
        else:
            msg = "PASSED"
        print("-------------------------------------------------------------------------------------------------------- ")
        print ("| TEST CASE  %s | %s  | distance: %s, speed: %s " %(idx +1, msg,tc[3], tc[2]))
        print("-------------------------------------------------------------------------------------------------------- ")

