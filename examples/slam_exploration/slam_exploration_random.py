import gym
from gym import wrappers
import gym_gazebo
import time
import numpy as np

EPISODES = 10000 #Maximum number of episodes
if __name__ == '__main__':
    env = gym.make('GazeboSlamExploration-v0')
    print("START TESTING")
    # run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()
    time.sleep(2)
    while not done:
        action = np.random.randint(0,26)
        obs, reward, done, info = env.step(action)
        # print("Reward ",reward)
        episode_reward += reward
    env.close()
    print("FINISHED TEST:\n Reward: ",str(episode_reward))