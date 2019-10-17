# the AI run on the straight and increasing the speed
import sys
import time
from time import sleep
import numpy as np
from math import ceil
import os
from datetime import datetime
import re
import pandas as pd
import sys_out

from beamngpy.sensors import Electrics
from create_road import create_road_shape
from beamngpy import BeamNGpy, Scenario, Road, Vehicle, setup_logging

from random import choice, randint, gauss, seed
from shapely.geometry import MultiPoint
from shapely.geometry import Point
import matplotlib.pyplot as plt
import random

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

#SPEED of car[0, 170]
def conver_kms_ms(km):
    ms = km * 5/18
    max_ms = 170 *5/18   #max speed in beamNG 170
    norminal_ms = ms / max_ms
    return ms, norminal_ms

def str_to_beamNG_TcFormat(tc):
    speed =  float(tc[1])   
    tod =  float(tc[2])
    testcase = (tc[0], speed, tod, tc[3])
    return testcase
    
 #Road_shape, speed_car, light, weather
#('straight', 113, 0.6086956521739131, 'rainy')
def Encode_TC(tc):
    shape = tc[0]
    #normalize shape 
    shape_nl = ROAD_SHAPE[shape]
    speed_car = tc[1]
    light = tc[2]
    weather = tc[3]
    weather_nl = WEAHTER[weather]
    speed_car, speed_nl  =  conver_kms_ms(speed_car)
    beamNG_tc = (shape, speed_car, light, weather)
    normalize_tc = (shape_nl, speed_nl, light, weather_nl)
    
   # normalize_tc = (shape_nl, speed_car, light, weather_nl)
    return beamNG_tc, normalize_tc

def get_poits_roads(pointBeamNG):
    #pointsR = bng.get_road_edges('beamngpy_road_road_test_000')
    left = list()
    right = list()
    middle = list()
    for point in pointBeamNG:
        left.append(point['left'])
        right.append(point['right'])
        middle.append(point['middle'])
    return left,middle, right
    
def get_xyCoordinate(listpoint):
    x = list()
    y = list()
    for point in listpoint:
        x.append(point[0])
        y.append(point[1])
    return x,y
    #print ("all points on road",pointsR)

def draw_polygon (left,middle,right):
    '''
    this function collects the points on the defined road and draws polygon to check the car inlane
    :param left:
    :param middle:
    :param right:
    :return: polygon the defined road
    '''
    x = list()
    y = list()
    x_left,y_left = get_xyCoordinate(left)
    x_middle,y_middle = get_xyCoordinate(middle)
    x_right,y_right = get_xyCoordinate(right)
    x = x_right
    y = y_right
    x.append(middle[len(middle) - 1][0])
    y.append(middle[len(middle) - 1][1])
    
    #print("x middle %s"%(middle))
    #print("x left reverse %s"%(x_left[::-1]))
    #print("type x %s"%(type(x)))
    #print("type x left reverse %s"%(type(x_left[::-1])))
    #print("middlte begin", middle[0][0])
    #print("middlte last", middle[len(middle) - 1][1])
    
    x = x + x_left[::-1]
    y = y + y_left[::-1]
    x.append(middle[0][0])
    y.append(middle[0][1])
    points_poly = list(zip(x,y))
    poly = MultiPoint(points_poly).convex_hull
    # plt.plot(x,y)
    # plt.show()
    return poly

def draw_polygon_end_road(shape_type):
    '''
    The function identify the ending location of the defined roads
    :param shape_type:
    :return: polygon of ending road (straight, left_curve, right_curve, two_curve)
    '''
    x = list()
    y = list()
    end_point = dict()
    
    end_straight = {
        'near_bottom_left': [382.461181640625, 796.644287109375 + 10, -0.0465356707572937],
        'near_bottom_right': [380.42352294921875, 796.5462646484375 - 10, -0.039158403873443604],
        'far_bottom_left': [382.68560791015625, 791.979736328125 + 10, -0.03985565900802612],
        'far_bottom_right': [380.64794921875, 791.8817138671875 - 10, -0.032478392124176025]
    }
    
    end_left_curve = {
        'near_bottom_left': [152.16220092773438, 696.0198974609375+10, -0.04270482063293457],
        'near_bottom_right':[152.1619873046875,  693.9798583984375 -10, -0.04193150997161865],
        'far_bottom_left':  [156.83218383789062, 696.0194091796875+10, -0.03657114505767822],
        'far_bottom_right': [156.83197021484375, 693.9793701171875-10, -0.035797834396362305]
    }
    
    end_right_curve = {
        'near_bottom_left': [621.6874389648438, 693.9872436523438 + 10, -0.04251587390899658],
        'near_bottom_right': [621.6880493164062, 696.0272827148438 - 10, -0.041700124740600586], 
        'far_bottom_left': [617.0173950195312, 693.9885864257812 + 10, -0.037297964096069336], 
        'far_bottom_right': [617.0180053710938, 696.0286254882812 - 10, -0.03648221492767334]
    }
    
    end_two_curve = {
        'near_bottom_left': [560.64892578125, 655.253173828125 + 10, -0.04449760913848877],
        'near_bottom_right': [558.609130859375, 655.22900390625 -10, -0.03864157199859619],
        'far_bottom_left': [560.7042236328125, 650.58349609375 + 10, -0.04082942008972168], 
        'far_bottom_right': [558.6644287109375, 650.559326171875 - 10, -0.0349733829498291]
    }

    if (shape_type == 0): 
        end_point = end_straight
    elif (shape_type == 0.5): 
        end_point = end_right_curve
    elif (shape_type == 0.6):
        end_point = end_left_curve
    elif (shape_type == 0.8):
        end_point = end_two_curve

    x.append(end_point['near_bottom_left'][0])
    x.append(end_point['near_bottom_right'][0])
    x.append(end_point['far_bottom_right'][0])
    x.append(end_point['far_bottom_left'][0])
    x.append(end_point['near_bottom_left'][0])
   
    y.append(end_point['near_bottom_left'][1])
    y.append(end_point['near_bottom_right'][1])
    y.append(end_point['far_bottom_right'][1])
    y.append(end_point['far_bottom_left'][1])
    y.append(end_point['near_bottom_left'][1])
    
    points_poly = list(zip(x,y))
    poly = MultiPoint(points_poly).convex_hull
    # plt.plot(x,y)
    # plt.show()
    return poly

def test_point_in_road (poly, point):
    if point.within(poly):
        return True
    else:
        return False

def save_sampele_experiement(number_sampels, road_shape):
    '''
    The function generate samples testcases and run it on beamNG research
    Theses results are used to verify the mode of GPC
    :param number_sampels:
    :param road_shape:
    :return: files which save testcases and their results
    '''
    beanNG_tc_list, normalize_tc_list = gen_testcase(number_sampels, road_shape)
    setup_logging()
    # results= list()
    beamNG_tc_tmp = list()
    normalize_tc_tmp = list()
    for i in range(number_sampels):
        result = run_TC_on_BeamNG(beanNG_tc_list[i])
        road_shape = beanNG_tc_list[i][0]
        speed = beanNG_tc_list[i][1]
        light = beanNG_tc_list[i][2]
        weather = beanNG_tc_list[i][3]
        beamNG_tc_tmp.append([road_shape, speed, light, weather, result])

        norl_road_shape = normalize_tc_list[i][0]
        norl_speed = normalize_tc_list[i][1]
        norl_light = normalize_tc_list[i][2]
        norl_weather = normalize_tc_list[i][3]
        normalize_tc_tmp.append([norl_road_shape, norl_speed, norl_light, norl_weather, result])

    beamNG_tc = pd.DataFrame(beamNG_tc_tmp)
    normalize_tc = pd.DataFrame(normalize_tc_tmp)
    beamNG_tc = beamNG_tc.set_axis(['Road_shape', 'speed', 'light',
                                    'weather', 'result'], axis=1, inplace=False)
    normalize_tc = normalize_tc.set_axis(['Road_shape', 'speed', 'light',
                                          'weather', 'result'], axis=1, inplace=False)

    # save result of testcase
    path_dir = os.getcwd()
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y__%H%M%S")
    file_beamNG = path_dir + "\store\\beamNG_" + road_shape + '_' + date_time + ".txt"
    file_norl_tc = path_dir + "\store\\tc_normallize_" + road_shape + '_' + date_time + ".txt"
    beamNG_tc.to_csv(file_beamNG)
    normalize_tc.to_csv(file_norl_tc)

def  run_TC_on_BeamNG(tc):
    '''
    This function run testcase which is encode in BeamNG format
    :param tc: input set of testcase as the beamNG format
    :return: [0,1] 0- failure, 1- pass
    '''
    print("\n\n\n Testcase:  ", tc)
    shape = ROAD_SHAPE[tc[0]]
    speed_car = tc[1]
    tod = tc[2]
    weather = tc[3]
    
    beamng = BeamNGpy('localhost', 64256, home='D:/BeamNGReasearch/Unlimited_Version/trunk')
    #name of scenario folder, and json file
    scenario = Scenario('asfault', 'road_test')
    
    road_a = Road('track_editor_A_center',one_way = False, looped=True)
    road_b = Road('track_editor_A_center',one_way = False, looped=True)
    orig = (382.42309233026646,525.0,0.01)

    # launch car position
    pos = (492.8909606933594, 524.2318115234375, 0.19860762357711792)
    dir = (0.999970555305481, -0.007486032787710428, -0.0016832546098157763 -90)
    
    #teleport position
    pos_tel = (381.88482666015625, 541.792236328125, 0.19991230964660645)    #turn right
    dir_tel = (0.15575499832630157, 0.9877752661705017, 0.006357199512422085 +180)

    # create the straight road, the car is firtly launched in the defined road to reach the speed of testcase
    roadSpeed = create_road_shape(1)
    #when the car reach the requirement speed, it is teleported to this road
    pointsRoad = create_road_shape(shape)

    script = []
    for i in range(len(pointsRoad)):
        node = (pointsRoad[i][0] + orig[0],  pointsRoad[i][1] + orig[1], orig[2], 7)
        script.append(node)
    #adding road
    road_a.nodes.extend(script)
    scenario.add_road(road_a)
    
    script_tmp = []
    for i in range(len(roadSpeed)):
        node = (roadSpeed[i][0] + orig[0],  roadSpeed[i][1] + orig[1], orig[2], 7)
        script_tmp.append(node)
    road_b.nodes.extend(script_tmp)
    scenario.add_road(road_b)

    vehicle = Vehicle('ego_vehicle', model='etk800', licence='AI')
    # Create an Electrics sensor and attach it to the vehicle
    electrics = Electrics()
    vehicle.attach_sensor('electrics', electrics)
    #adding vehicle
    scenario.add_vehicle(vehicle, pos=pos, rot=dir)
    scenario.make(beamng)
    bng = beamng.open(launch=True)
    inlane = True
    
    poly_end = draw_polygon_end_road(shape)
    
    try:
        bng.load_scenario(scenario)
        bng.start_scenario()
        bng.set_weather_preset(weather, time = 1)
        bng.set_tod(tod)
        vehicle.ai_set_speed(speed=speed_car,mode ='set')
        vehicle.ai_set_mode('span')
        vehicle.update_vehicle()
        sensors = bng.poll_sensors(vehicle)
        vehicle.update_vehicle()
        sensors = bng.poll_sensors(vehicle)
        #draw polygon for the road to check the car in lane of the road
        pointsR = bng.get_road_edges('beamngpy_road_road_test_000')
        left, middle, right = get_poits_roads(pointsR)
        poly = draw_polygon(left, middle, right)

        flag_teleport = False
        #launch the car to reach the speed of testcase
        for _ in range(400):
            time.sleep(0.1)
            vehicle.update_vehicle()  # Synchs the vehicle's "state" variable with the simulator
            sensors = bng.poll_sensors(vehicle)  # Polls the data of all sensors attached to the vehicle
            speed = sensors['electrics']['values']['wheelspeed']
            #print ("speed from sensor ", speed)
            #print ("speed car", speed_car)
            #print([vehicle1.state['pos'],vehicle1.state['dir']])
            if ceil(speed) >= int(speed_car):                            #change speed here
                bng.teleport_vehicle(vehicle,pos_tel,dir_tel )
                print("\n \nTeleport car to new location \n")
                flag_teleport = True
                break
        if not flag_teleport:
            bng.stop_scenario()
        for _ in range(450):
            time.sleep(0.1)
            vehicle.update_vehicle()  # Synchs the vehicle's "state" variable with the simulator
            #sensors = bng.poll_sensors(vehicle)  # Polls the data of all sensors attached to the vehicle
            #box = vehicle.get_bbox()
            inlane = test_point_in_road(poly,Point(vehicle.state['pos'][0], vehicle.state['pos'][1]))
            endPoint = test_point_in_road(poly_end,Point(vehicle.state['pos'][0], vehicle.state['pos'][1]))
            if not inlane:
                bng.stop_scenario()
                return 0
            elif (inlane and endPoint ):
                bng.stop_scenario()
                return 1
    finally:
        bng.close()

def create_sample_AL(num_pool, num_init_sample, road_shape):
    '''
    The function creates pool samples of testcases, and takes small initial testcases from the pool
    to train the model for active learning process. the initial testcases are run in BeamNG.research
    to get the result
    :param num_pool: number of pool
    :param num_init_sample: number of testcases which is used in initial step of active learning
    :param road_shape: road shape
    :return: X_initial, result_init, pool_X, pool_X_bng
    '''
    sys_out.print_title("Run initial samples on BeamNG to build initial training model")
    beanNG_tc_list, normalize_tc_list  = gen_testcase(num_pool, road_shape)
    setup_logging()

    result_init = list()
    list_idx = random.sample([i for i in range(num_pool)],num_init_sample)
    X_initial = [normalize_tc_list[idx] for idx in list_idx ]

    result_init = [randint(0,1) for i in range(num_init_sample)] #todo command this to run on real env

    
    #uncomment this part to get the result of testcase from initial
    # for i in range(num_init_sample): #todo replace initial random result with BeamNG result
    #     result = run_TC_on_BeamNG(beanNG_tc_list[list_idx[i]])
    #     result_init.append(result)

    #delete initial testcases in the pool 
    pool_X = np.delete(np.asarray(normalize_tc_list),  list_idx, axis=0)
    pool_X_bng = np.delete(np.asarray(beanNG_tc_list),  list_idx, axis=0)
    return X_initial, result_init, pool_X, pool_X_bng


def gen_testcase(numTC, road_type='None'):
    '''
    Generate the testscase automatically, each testcase is set of the attributes
    (Road_shape,speed,light,weather,result):
        BeamNG_testcase: left_curve,33.611111111111114,0.5217391304347826,foggy_night
        normalize_testcase: 0.6,0.711764705882353,0.5217391304347826,0.8
    :param numTC: number of testcases which are generated
    :param road_type: type of road : straight, left_curve, right_curve, two_curve
    :return: list of testcases which are able to run on beamNG, and testcases which is encode for using in algorithms
    '''
    print("The testcases are generate with fields: \n\n \
          Road_shape, speed_car, light, weather \n\n")
    TCs = list()
    road_shape = ['straight', 'right_curve', 'left_curve', 'two_curve']
    weather_cond = ['sunny', 'sunny_noon', 'sunny_evening', 'cloudy_evening',
                    'rainy', 'foggy_morning', 'foggy_night']

    for i in range(numTC):
        if road_type == 'None':
            road = choice(road_shape)
        else:
            road = road_type
        # traff_sign = choice(traff_sign_speed)
        # speed_rad = randint(10,200)
        speed_rad = randint(30, 170)  # max speed of car in beamNG
        tod = np.linspace(0, 1, 24)
        light = choice(tod)
        weather = choice(weather_cond)
        tc = (road, speed_rad, light, weather)
        TCs.append(tc)

    # encode TC to run_on BeamNG and run Gaussian process
    beanNG_tc_list = list()
    normalize_tc_list = list()
    for tc in TCs:
        b_tc, nor_tc = Encode_TC(tc)
        beanNG_tc_list.append(b_tc)

        print("BeamNGTC", b_tc)

        normalize_tc_list.append(nor_tc)
    return beanNG_tc_list, normalize_tc_list


def main(num_tc, road_shape):
    #create_sample_AL(100,20, 'left_curve')
    # save_sampele_experiement(15, 'left_curve')
    # save_sampele_experiement(15, 'left_curve')
    # save_sampele_experiement(20, 'left_curve')
    #save_sampele_experiement(15, 'right_curve')
    #save_sampele_experiement(15, 'two_curve')

    beanNG_tc_list, normalize_tc_list = gen_testcase(int(num_tc), road_shape)

    for tc in beanNG_tc_list:
        rs = run_TC_on_BeamNG(tc)
        if rs == 0:
            sys_out.result("FAILED")
        elif rs == 1:
            sys_out.result("PASSED")
        else:
            sys_out.result("NONE")
        #print("\n Result is %s" %(run_TC_on_BeamNG(tc)))

  
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
