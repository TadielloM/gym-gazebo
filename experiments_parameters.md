# Experiment parameters

## 1

Maximum time per episode = 60s
Discount factor = 0.95
Learning rate = 0.001
self.epsilon = 0.05
Batch size = 5
Memory size = 100
Learning starts = 30
Target update frequency = 20

## 2
### Results: Q value nan (see going trough one tunnel)

Maximum time per episode = 60s
Discount factor = 0.9
Learning rate = 0.001
self.epsilon = 0.05
Batch size = 5
Memory size = 100
Learning starts = 30
Target update frequency = 25
Octomap Resolution = 0.1
octomap sensor_range = 10


## 3

Maximum time per episode = (60*1.5)s
max episodes = 801
Discount factor = 0.9
Learning rate = 0.001
self.epsilon = 0.1
Batch size = 5
Memory size = 100
Learning starts = 40
Target update frequency = 25
Octomap Resolution = 0.2
Octomap and Sensor range = 200


## 4 (After big refactor sensor and octomap) 22/09 16:34 
### Results: Crashed at episode 524 becouse over Maximum number of points

Maximum number of points = 100000
T_FACTOR = 5
Discount factor = 0.9
Learning rate = 0.001
self.epsilon = 0.1
Batch size = 5
Memory size = 100
Learning starts = 40
Target update frequency = 25

Octomap Resolution = 0.2
octomap sensor_range = 150

## 5 23/09 11:48
### Results: Learnt to go to big zones, finished in around 100 episodes reach maximum number of points


Maximum number of points = 100000
Maximum time per episode = 60*3
Discount factor = 0.9
Learning rate = 0.001
self.epsilon = 0.1
Batch size = 5
Memory size = 100
Learning starts = 40
Target update frequency = 25
Octomap Resolution = 0.2
Octomap and Sensor range = 65

## 6 23/09 17:56
### Results: It wasn't really converging (bad results)

Maximum number of points = 125000
Maximum time per episode = 60*3
Discount factor = 0.9
Learning rate = 0.001
self.epsilon = 0.1
Batch size = 4
Memory size = 100
Learning starts = 40
Target update frequency = 25
Octomap Resolution = 0.2
Octomap and Sensor range = 65

## 7 24/09 21:55 (Using double DQN and epsilon decay linearly)
### Results: Seems was not learning, stopped at episode 439

Maximum number of points = 125000
Maximum time per episode = 60*3
Discount factor = 0.9
Learning rate = 0.001
self.epsilon = 0.1
Initial Epsilon = 1.0
Final Epsilon = 0.02
Batch size = 4
Memory size = 100
Learning starts = 40
Target update frequency = 25
Octomap Resolution = 0.2
Octomap and Sensor range = 65

## 8 25/09 21:55 (Using double DQN and epsilon decay inversely proportional)
### Results: (Was not training becouse learning start need to be minor than memory size)

Maximum number of points = 125000
Maximum time per episode = 60*3
Discount factor = 0.9
Learning rate = 0.001
self.epsilon = 0.1
Initial Epsilon = 1.0
Final Epsilon = 0.1
Batch size = 4
Memory size = 40
Learning starts = 41
Target update frequency = 25
Octomap Resolution = 0.2
Octomap and Sensor range = 65

## 9 26/09 11:48 (Using double DQN and epsilon decay inversely proportional)
### Results (nothing learnt stop at 450)

Maximum number of points = 125000
Maximum time per episode = 60*3
Discount factor = 0.9
Learning rate = 0.001
self.epsilon = 0.1
Initial Epsilon = 1.0
Final Epsilon = 0.1
Batch size = 4
Memory size = 40
Learning starts = 20
Target update frequency = 25
Octomap Resolution = 0.2
Octomap and Sensor range = 65

## 10 27/09 14:49 (Using double DQN and epsilon decay inversely proportional and stay still action and negative reward if "going trough a wall")
### Results Stopped for error at episdde 300, too early to say

Maximum number of points = 125000
Maximum time per episode = 60*3
Discount factor = 0.9
Learning rate = 0.001
self.epsilon = 0.1
Initial Epsilon = 1.0
Final Epsilon = 0.1
Batch size = 4
Memory size = 40
Learning starts = 20
Target update frequency = 25
Octomap Resolution = 0.2
Octomap and Sensor range = 65

## 11 28/09 10:06  (Using double DQN and epsilon decay inversely proportional and stay still action and negative reward if "going trough a wall")
### Results (Does not learn good, q value stacked to q_value: [1680.81140137])


Maximum number of points = 125000
Maximum time per episode = 60*3
Discount factor = 0.9
Learning rate = 0.001
self.epsilon = 0.1
Initial Epsilon = 1.0
Final Epsilon = 0.1
Batch size = 4
Memory size = 40
Learning starts = 20
Target update frequency = 25
Octomap Resolution = 0.2
Octomap and Sensor range = 65


## 12  (Using double DQN and epsilon decay inversely proportional and stay still action and negative reward if "going trough a wall")
### Results stuck to something during learning


Maximum number of points = 125000
Maximum time per episode = 60*3
Discount factor = 0.9
Learning rate = 0.0005
self.epsilon = 0.1
Initial Epsilon = 1.0
Final Epsilon = 0.1
Batch size = 4
Memory size = 40
Learning starts = 20
Target update frequency = 10
Octomap Resolution = 0.2
Octomap and Sensor range = 65

## 13 01/10 09:21 (Using double DQN and epsilon decay inversely proportional and stay still action and negative reward if "going trough a wall") Removed time limit only number of point for end the episode
### Results 


Maximum number of points = 125000
Maximum time per episode = No time limit
Discount factor = 0.9
Learning rate = 0.0005
Initial Epsilon = 1.0
Final Epsilon = 0.1
Batch size = 4
Memory size = 40
Learning starts = 20
Target update frequency = 10
Octomap Resolution = 0.2
Octomap and Sensor range = 65

## 14 (Using rrllib default configuration with double DQN and Dueling DQN greedy epsilon )

Maximum number of points = 125000
Maximum time per episode = No time limit
Number of maximum episodes = 101
Discount factor = 0.99
Learning rate = 0.0005
Timesteps for decreasing Epsilon = 10000
Final Epsilon = 0.02
Initial Epsilon = 1.0
Type of exploration = Epsilon Greedy
Batch size = 2
Learning starts = 20
Target update frequency = 500
Octomap Resolution = 0.2 m
Octomap and Sensor range = 65 m

## 15 (Using rrllib default configuration with double DQN and Dueling DQN greedy epsilon )

Maximum number of points = 125000
Maximum time per episode = No time limit
Number of maximum episodes = 501
Discount factor = 0.99
Learning rate = 0.0005
Timesteps for decreasing Epsilon = 10000
Final Epsilon = 0.02
Initial Epsilon = 1.0
Type of exploration = Epsilon Greedy
Batch size = 2
Learning starts = 20
Target update frequency = 500
Octomap Resolution = 0.2 m
Octomap and Sensor range = 65 m

## L' ultimo test per sempre


Maximum number of points = 125000
Maximum time per episode = infinito
Discount factor = 0.99
Learning rate = 0.0005
Initial Epsilon = 1.0
Final Epsilon = default
Batch size = 2
Memory size = 1000
Learning starts = 20
Target update frequency = 500
Octomap Resolution = 0.2
Octomap and Sensor range = 65