# Boundary Identification

The project is to find the boundary of testcases for testing self-driving car. 
The scope of the topic is searching based software.
Each testcase has parameters (road shape, speed of car, light, weather condition)

The first step is sampling testcase to choose the testcases that potentially identify the boundary. 
The boundary of the testcases separates safe testcases from unsafe testcases.
The safe testcase is the testcase that can be on track when running on BeamNG simulation.

The simulation is start as below:
 1. The road is defined in a particular shape and distance.
 2. Scenario for a testcase includes: speed of car, car type, light of environment, weather condition
 3. Preconfiguation the testcase with paramters as above
 4. Open BeamNG simulation and launch the car with above conditions. 
 5. The car is run automatically with AI mode in BeamNG simulation.
 6. During running the car in AI mode, the simulation can checks the position of the vehicle from the sensor of the vehicle and return the position of the vehicle.
 7. The application measure the result safe or unsafe of a testcase base on the position of car on the road.
 
 The approach to find the boundary:
 
 - Sampling data with active learning method: using gaussian process algorithm and entropy selected strategy
 - Identify the boundary: using cluster algorim KNN
 

