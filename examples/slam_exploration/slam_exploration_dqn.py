#!/usr/bin/env python
import gym
from gym import wrappers
import gym_gazebo
import time
import numpy as np
from collections import deque
import random
import keras
from keras.optimizers import Adam
import tensorflow as tf
import sys, signal, os
from datetime import datetime
import matplotlib.pyplot as plt

EPISODES = 10000 #Maximum number of episodes

class DQNAgent:
    def __init__(self, obs_space, action_space, log_path, name = "Agent-0"):
        
        #store observation and action space (size)
        self.obs_space = obs_space
        self.action_space = action_space.n
        print("The obs space is: ", obs_space.shape)
        print("The action space is: ", action_space.n)
        #Hyper parameters
        self.discount_factor = 0.9
        self.learning_rate = 0.0005
        self.epsilon = 0.1
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.1
        self.batch_size = 4
        self.memory_size = 40
        self.learning_starts = 20
        self.target_update_frequency = 10
        # self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_path)

        #Memory buffer implemented as deque collection
        self.buffer = deque(maxlen=self.memory_size)

        #Create main network and target network (using build_model defined below)
        self.model = self.build_model()
        self.target_model = self.build_model()

        #Initialize target network
        self.update_target_model()
        

    def build_model(self):
        print("The input model need to be of the shape: ", self.obs_space.shape)
        inputs = tf.keras.layers.Input(shape=self.obs_space.shape, name="observations")
        shared = tf.keras.layers.Convolution1D(64,1,activation='relu')(inputs)
        shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.Convolution1D(64,1,activation='relu')(shared)
        shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.Convolution1D(128, 1, activation='relu')(shared)
        shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.Convolution1D(1024, 1, activation='relu')(shared)
        shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.MaxPooling1D(pool_size=self.obs_space.shape[0])(shared)
        shared = tf.keras.layers.Flatten()(shared)
        shared = tf.keras.layers.Dense(512, activation='relu')(shared)
        shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.Dense(256, activation='relu')(shared)
        shared = tf.keras.layers.Dense(128, activation='relu')(shared)

        action_probs = tf.keras.layers.Dense(name="action_prob", units=self.action_space, activation='linear')(shared)
        
        model = tf.keras.Model(inputs=inputs , outputs=action_probs)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        # self.register_variables(self.base_model.variables)
        model.summary()
        print("Model layers shape: ")
        for layer in model.layers:
            print(layer.output_shape)

        return model 

    #After some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action_training(self, state, episode):
        #TODO: Modify exloration within time (explore more at the start of the training)
        #Epsilon greedy technique
        # epsilon = self.initial_epsilon-((self.initial_epsilon/EPISODES)*(episode+1))
        epsilon = self.initial_epsilon/(episode+1)
        if( epsilon <self.final_epsilon):
            epsilon = self.final_epsilon
        if np.random.random() < epsilon:
            # print("Exploration")
            action = np.random.randint(0,self.action_space)
        else:
            # print("Exploitation")
            # print("State",type(state),state.shape,state)
            Qs = self.model.predict(state.reshape(1,env.observation_space.shape[0],env.observation_space.shape[1]))
            action = np.argmax(Qs)
        return action

    def get_action_random(self,state):
        # Qs = self.model.predict(state.reshape(1,env.observation_space.shape[0],env.observation_space.shape[1]))
        action = np.random.randint(0,self.action_space)
        return action

    def get_action(self, state):
        # print("State",type(state),state.shape,state)
        Qs = self.model.predict(state.reshape(1,env.observation_space.shape[0],env.observation_space.shape[1]))
        action = np.argmax(Qs)
        return action

    def train_model(self):
        if len(self.buffer) < self.learning_starts:
            return
        batch_size = min(self.batch_size, len(self.buffer)) #Train on at most as many samples as you have in memory
        mini_batch = random.sample(self.buffer, batch_size) #Uniformly sample the memory buffer
        actual_state = np.zeros((batch_size, self.obs_space.shape[0],self.obs_space.shape[1])) #batch_size by state_size two-dimensional array
        new_state = np.zeros((batch_size, self.obs_space.shape[0],self.obs_space.shape[1])) #Same as above, but used for the target network
        action, reward, done = [],[],[]
         
        for i in range(self.batch_size):
            actual_state[i]=mini_batch[i][0] #Allocate s(i) to the network input array from iteration i in the batch
            action.append(mini_batch[i][1]) #Store a(i)
            reward.append(mini_batch[i][2]) #Store r(i)
            new_state[i]=mini_batch[i][3] #Allocate s'(i) for the target network array from iteration i in the batch
            done.append(mini_batch[i][4]) 
            #update y based on the last rewards and done variable
            
        q_val = self.model.predict(actual_state) #Generate target values for training the inner loop network using the network model
        q_val_next_state = self.model.predict(new_state) #for DDQN
        target_val = self.target_model.predict(new_state) #Generate the target values for training the outer loop target network
        # print("Target_val: ",type(target_val), target_val.shape)
        # print("target_val: ",target_val)

        for i in range(self.batch_size):
            if done[i] == True:
                q_val[i][action[i]]= reward[i]
            else:
                # print("amax: ",np.amax(target_val[i]))
                # q_val[i][action[i]] = reward[i] + self.discount_factor*np.amax(target_val[i]) #DQN
                q_val[i][action[i]] = reward[i] + self.discount_factor*target_val[i][np.argmax(q_val_next_state[i])] #Double DQN

            # print("q_val_i :",q_val[i])
        # print("Arrivo a trainare")
        # self.model.fit(x=actual_state, y=q_val, batch_size=self.batch_size, epochs=1, verbose=0, callbacks=[self.tensorboard_callback])
        self.model.fit(x=actual_state, y=q_val, batch_size=self.batch_size, epochs=1, verbose=0,)




if __name__ == '__main__':
    env = gym.make('GazeboSlamExploration-v0')

    
    path = os.path.expanduser('~/saved_models_and_log/')
    if not os.path.exists(path):
        os.makedirs(path)
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M")
    model_path = path+"GazeboSlamExploration-v0_"+dt_string
    log_path = path+"/logs/GazeboSlamExploration-v0_"+dt_string

    file_writer = tf.summary.create_file_writer(log_path)

    print(model_path)

    agent = DQNAgent(env.observation_space, env.action_space, log_path)

    #list of scores for each episode
    scores, episodes = [], [] 

    def handler(signum, frame):
        print('Sigint detected, closing environment: ', signum)
        env.close()
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)
    
    start_time = time.time()
    

    max_q = np.zeros((EPISODES, agent.batch_size))
    max_q_mean = np.zeros((EPISODES,1))
    total_reward = 0
    is_training = False
    if is_training :
        for e in range(EPISODES):
            print("Starting episode ",e)
            done = False
            cumulated_reward=0
            state = env.reset()
            # print("In DQN State: ", type(state), state.shape, state)
            
            # For plotting avg q_value for "batch like" input
            batch_size = min(agent.batch_size, len(agent.buffer))
            mini_batch = random.sample(agent.buffer, batch_size)
            test_states = np.zeros((agent.batch_size, agent.obs_space.shape[0],agent.obs_space.shape[1]))
            tmp = agent.model.predict(test_states)
            max_q[e][:] = np.max(tmp, axis=1)
            max_q_mean[e] = np.mean(max_q[e][:])
            episode_iterations = 0
            while not done:
                episode_iterations+=1
                # print("get_action")
                action = agent.get_action_training(state,e)
                # print("computing step")
                next_state, reward, done, info = env.step(action)
                # print("appending buffer")
                agent.buffer.append((state,action,reward,next_state,done))
                # print("training model")
                agent.train_model()
                
                #store episode reward
                cumulated_reward += reward 
                #update state
                state = next_state

                if done:
                    if e % agent.target_update_frequency == 0:
                        agent.update_target_model()

                    scores.append(cumulated_reward)
                    episodes.append(e)
                    if (e%10 == 0):
                        agent.model.save(model_path)
                        print("Model saved in path: ", model_path)
                    total_reward += cumulated_reward
                    avg_score_ten_ep = 0
                    if(e>10):
                        for i in range(10):
                            avg_score_ten_ep += scores[-i]
                    avg_score_ten_ep = avg_score_ten_ep/10
                    with file_writer.as_default():
                        tf.summary.scalar("score", cumulated_reward, step=e)
                        tf.summary.scalar("avg reward during episode", cumulated_reward/episode_iterations, step=e)
                        tf.summary.scalar("avg_score", total_reward/(e+1), step=e)
                        tf.summary.scalar("avg score last ten episodes",avg_score_ten_ep, step=e)
                        tf.summary.scalar("Q_val", float(max_q_mean[e]), step=e)
                        tf.summary.scalar("Iteration_per_episode", episode_iterations ,step=e)
                        file_writer.flush()
                    print("episode: {}/{} score: {} avg_score: {:.2f}, q_value: {} memory length: {}".format(e, EPISODES, cumulated_reward, total_reward/(e+1), max_q_mean[e] , len(agent.buffer)))

            


    else:
        print("START TESTING")
        checkpoint_path = "/home/tadiellomatteo/saved_models_and_log/GazeboSlamExploration-v0_30_09_2020__20_16"
        model = keras.models.load_model(checkpoint_path)
        # run until episode ends
        episode_reward = 0
        done = False
        obs = env.reset()
        time.sleep(2)
        while not done:
            action = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            # print("Reward ",reward)
            episode_reward += reward
        env.close()
        print("FINISHED TEST:\n Reward: ",str(episode_reward))