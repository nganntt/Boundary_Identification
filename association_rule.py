# the AI run on the straight and increasing the speed

import numpy as np
from math import ceil
from random import choice, randint, gauss, seed
from shapely.geometry import MultiPoint




import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

ROAD_SHAPE = {
    'straight': 0,
    'right_curve': 0.5,
    'left_curve': 0.6,
    'two_curve': 0.8
}

WEAHTER = {
    'sunny': 0,
    'sunny_noon':0.2,
    'sunny_evening': 0.35,
    'cloudy_evening': 0.35,
    'rainy': 0.5,
    'foggy_morning': 0.7,
    'foggy_night':0.8
}

TIMEOFDAY = {
    '1_to_5': 0.1,        
    '5_to_7': 0.5,        
    '7_to_9': 0.7,        
    '9_to_10':0.9,       
    '10_to_14': 1,       
    '14_to_16': 0.8,      
    '16_to_18': 0.6,      
    '18_to_19': 0.4,      
    '19_to_20': 0.3,      
    '20_to_21': 0.2,     
    '21_to_24': 0        
}

def encode_speed(speed):
    """
    Distance encode for decision tree:
        speed_lt_20 in  [0,20]
        speed_bw_20_40 in  [20,40]
        speed_bw_40_60 in  [40,60]
        speed_bw_60_80 in  [60,80]
        speed_bw_80_100 in  [80,100]
        speed_bw_100_120 in  [100,120]
        speed_bw_120_140 in  [120,140]
        speed_bw_140_160 in  [140,160]
        speed_bw_160_180 in  [160,180]
        speed_gt_180  [180,200]
    """
    if speed < 20:
        return 'speed_lt_20'
    if 20<= speed < 40:
        return 'speed_bw_20_40'
    if 40<= speed < 60:
        return 'speed_bw_40_60'
        
    if 60<= speed < 80:
        return 'speed_bw_60_80'
    if 80<= speed < 100:
        return 'speed_bw_80_100'
    if 100<= speed < 120:
        return 'speed_bw_100_120'
    if 120<= speed < 140:
        return 'speed_bw_120_140'
    if 140<= speed < 160:
        return 'speed_bw_140_160'
    if 160<= speed < 180:
        return 'speed_bw_160_180'
    if speed >= 180:
        return 'speed_gt_180'




def conv_TOD(tod):
    str_tod = ""
    for key, value in TIMEOFDAY.items():    
        if value == round(tod,1):
            str_tod = key
    conv_value = ''
    if (str_tod == '1_to_5') or (str_tod == '21_to_24') or (str_tod == '20_to_21') :
        conv_value = 'light_night'
    elif (str_tod == '5_to_7') or (str_tod == '18_to_19') or (str_tod == '19_to_20'):
        conv_value = 'light_weak'
    elif (str_tod == '7_to_9') or (str_tod == '16_to_18') :
        conv_value = 'light_medium'
    elif (str_tod == '10_to_14') or (str_tod == '9_to_10') or (str_tod == '14_to_16'):
        conv_value = 'light_strong'
    return conv_value

def decode_testcase_in_boundary_list(boundary_tc):
    """
    decode testcase from a list tuples of testcase normalized
    normalize_tc = (shape_road_nl, speed_car, light, weather_nl)
    ex:  np.array([0.6,     0.565 ,   0.56521739,     0.35])
    """
    testcases_association_rule = list()
    beamNG_list = list()
    for i in range(len(boundary_tc)) :
        road_shape1 = boundary_tc[i][0]
        speed1      = boundary_tc[i][1]
        light1      = boundary_tc[i][2]
        weather1    = boundary_tc[i][3]
        road_shape2 = boundary_tc[i][4]
        speed2      = boundary_tc[i][5]
        light2      = boundary_tc[i][6]
        weather2    = boundary_tc[i][7]

        # convert road_shape
        road_shape1_str = ""
        road_shape2_str = ""
        for key, value in ROAD_SHAPE.items():
            if value == road_shape1:
                road_shape1_str = key
            if value == road_shape2:
                road_shape2_str = key

        # convert light
        light1_str = conv_TOD(light1)
        light2_str = conv_TOD(light2)

        # for key, value in TIMEOFDAY.items():
            # if value == round(light1,1):
                # light1_str = key
            # if value == round(light2,1):
                # light2_str = key

         # convert weather
        weather1_str = ""
        weather2_str = ""
        for key, value in WEAHTER.items():
            if value == weather1:
                weather1_str = key
            if value == weather2:
                weather2_str = key

        #conver to km/h
        # speed1_km = round(speed1 * 170 *5/18 *3.6, 2 ) #normalize divide for 170 in scenarios
        # speed2_km = round(speed2 * 170*5/18 *3.6, 2)
        speed1_km = encode_speed(ceil(speed1 * 170 *5/18 *3.6))#normalize divide for 170 in scenarios
        speed2_km = encode_speed(ceil(speed2 * 170 *5/18 *3.6))

        state1 = 'safe'
        state2 = 'unsafe'
        tc1 = np.array([road_shape1_str, speed1_km, light1_str, weather1_str, state1])
        tc2 = np.array([road_shape2_str, speed2_km, light2_str, weather2_str, state2])
        beamNG_list.append([*tc1,*tc2])
        testcases_association_rule.append(tc1)
        testcases_association_rule.append(tc2)
    return testcases_association_rule, beamNG_list



#1, ['left_curve' 'speed_bw_140_160' 'light_weak' 'cloudy_evening']
# 2, ['right_curve' 'speed_bw_160_180' 'light_medium' 'cloudy_evening']
def convert_dataFrame_rs(testcases):
    """
    This function convers data from a list of testcases to dataFrame for decision tree
    this function return a table of data
    """
    data_dt = []
    for item in testcases:
        #road_shape = item[0]
        speed      = item[1] 
        #light      = item[2]
        #weather    = item[3] 
        state      = item[4]
        #data_dt.append((road_shape, speed, light, weather))
        data_dt.append((speed, state))
       # print(item)
   
    #conver to dataFrame 
    dfObj_tmp = pd.DataFrame(data_dt) 
   
  
    dfObj = dfObj_tmp.set_axis(['speed', 'state'], axis=1, inplace=False)
 
    one_hot_data_state = pd.get_dummies(dfObj['state'])
    one_hot_data_speed = pd.get_dummies(dfObj['speed'])
    print("\n safe and unsafe \n", one_hot_data_state)
    
    #data has format
    #dist_bw_20_30  dist_bw_30_40  dist_gt_40  dist_lt_20  speed_bw_20_40  speed_bw_40_60  speed_gt_60  state
    data_training_Apriori = pd.concat([one_hot_data_state, one_hot_data_speed],axis=1,sort=False)

    # data_training_DT = check_dataFrame(data_training_DT_temp)
    # print ("\n Convert data to dataframe to train DT \n", data_training_DT )
    
    print(data_training_Apriori)
    
    return data_training_Apriori


##
#convert_dataFrame(dtc)
def apriori_speed_rs (dataset):
    te = TransactionEncoder()
    
    te_ary = te.fit(dataset).transform(dataset)
    #df = pd.DataFrame(te_ary, columns=te.columns_)
    df = convert_dataFrame_rs(dataset)
    print("Head data frame for Apriori algorithm", df.head())
    frequent_itemsets = apriori(df, min_support = 0.1, use_colnames=True)
    #adding length
    #frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    #select itemsets which satisfies the conditions
    # a = frequent_itemsets[ (frequent_itemsets['length'] == 2) &
                   # (frequent_itemsets['support'] >= 0.2) ]
    print (frequent_itemsets)
    # print ("\n \n ", a)
    return frequent_itemsets

def rules_frequenItemsets_rs(frequent_itemsets):
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
    #rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    print ("Rule", rules)
 
