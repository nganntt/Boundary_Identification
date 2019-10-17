
import os
from datetime import datetime
import pandas as pd
import numpy as np


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

