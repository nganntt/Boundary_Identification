from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from shapely.geometry import MultiPoint
from shapely.geometry.polygon import LinearRing

NODES_STRAIGHT   = [[0, 0], [0, 100], [0, 300]]
NODES_TURN_LEFTS = [[0, 0],[0, 50],[-20, 150], [-150, 170],[-250, 170]]
#NODES_TURN_LEFTS = [[250, 2],[260, 80],[260, 150], [150, 180], [0,180]]
NODES_TURN_RIGHT = [[0, 0], [0, 50], [20,150], [150, 170], [250,170] ] 
NODES_SINE       = [[0, 0], [0, 35], [5, 75], [25, 100], [80,75],[150,50], [175,100],[175,150]]
NODE_TEMP        = [[100,0], [300,0], [500,0], [1000,0], [2000,0]]

def create_road_shape(shape):
    """
        shape = 0 : NODES_STRAIGHT
        shape = 0.6 : NODES_TURN_LEFTS
        shape = 0.5 : NODES_TURN_RIGHT
        shape = 0.8 : NODES_SINE
    """
    listPoints = []
    if (shape == 0):
        listPoints = NODES_STRAIGHT
    elif (shape == 0.6):
        listPoints = NODES_TURN_LEFTS
    elif (shape == 0.5):
        listPoints = NODES_TURN_RIGHT
    elif (shape == 0.8):
        listPoints = NODES_SINE
    elif (shape == 1):     #road to reach the speed
        listPoints = NODE_TEMP

    nodes = np.array(listPoints)
    x = nodes[:,0]
    y = nodes[:,1]

    tck,u     = interpolate.splprep( [x,y] ,s=0, k=2)
    xnew,ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck,der = 0)
    points = list(zip(xnew,ynew))
    # plt.plot( x,y,'o' , xnew ,ynew )
    # plt.legend( [ 'data' , 'spline'] )
    # plt.axis( [ x.min() - 20 , x.max() + 30 , y.min() - 30 , y.max() + 20 ] )
    # plt.show()
    return points

def polygon_road(points):
    #coords is a list of (x, y) tuples
    poly = MultiPoint(coords).convex_hull
    ring = LinearRing([(0, 0), (1, 1), (1, 0)])
    
# create_road_shape(0)
# create_road_shape(0.5)
# create_road_shape(0.6)
#create_road_shape(0.8)

#create_road_shape(1)

