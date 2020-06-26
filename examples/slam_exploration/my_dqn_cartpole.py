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
from keras.models import Sequential

EPISODES = 1000 #Maximum number of episodes


class DQNAgent:
    def __init__(self, obs_space, action_space, name = "Agent-0"):
        
        #store observation and action space (size)
        self.obs_space = obs_space.shape[0]
        self.action_space = action_space.n
        print("The obs space is: ", obs_space.shape)
        print("The action space is: ", action_space.n)
        #Hyper parameters
        self.discount_factor = 0.95
        self.learning_rate = 0.001
        self.epsilon = 0.02
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.02
        self.batch_size = 4
        self.memory_size = 100
        self.learning_starts = 64
        self.target_update_frequency = 5
        self.test_state_no = 10000
        #Memory buffer implemented as deque collection
        self.buffer = deque(maxlen=self.memory_size)

        #Create main network and target network (using build_model defined below)
        self.model = self.build_model()
        self.target_model = self.build_model()

        #Initialize target network
        self.update_target_model()
        

    def build_model(self):
        print("The input model need to be of the shape: ",  type(self.obs_space), self.obs_space)
        inputs = tf.keras.layers.Input(shape=self.obs_space, name="observations")
        shared = tf.keras.layers.Dense(64, activation='relu')(inputs)
        # shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.Dense(128, activation='relu')(shared)

        shared = tf.keras.layers.Dense(name="h1", units=32, activation='relu')(shared)

        action_probs = tf.keras.layers.Dense(name="p", units=self.action_space, activation='linear')(shared)
        
        model = tf.keras.Model(inputs=inputs , outputs=action_probs)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        # self.register_variables(self.base_model.variables)
        model.summary()
        return model 

    #After some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state,episode):
        #TODO: Modify exloration within time (explore more at the start of the training)
        epsilon = self.initial_epsilon/(episode+1) #-((self.initial_epsilon/EPISODES)*(episode+1))
        if( epsilon <self.final_epsilon):
            epsilon = self.final_epsilon

        # print("Epsilon",epsilon)
        if np.random.random() < epsilon:
            # print("Exploration")
            action = np.random.randint(0,self.action_space)
        else:
            # print("Exploitation")
            # print("State",type(state),state.shape,state)
            Qs = self.model.predict(state.reshape(1,self.obs_space))
            action = np.argmax(Qs)
        return action

    def train_model(self):
        if len(self.buffer) < self.learning_starts:
            return
        batch_size = min(self.batch_size, len(self.buffer)) #Train on at most as many samples as you have in memory
        mini_batch = random.sample(self.buffer, batch_size) #Uniformly sample the memory buffer
        actual_state = np.zeros((batch_size, self.obs_space)) #batch_size by state_size two-dimensional array
        new_state = np.zeros((batch_size, self.obs_space)) #Same as above, but used for the target network
        actions, rewards, dones = [],[],[]
         
        for i in range(self.batch_size):
            actual_state[i]=mini_batch[i][0] #Allocate s(i) to the network input array from iteration i in the batch
            actions.append(mini_batch[i][1]) #Store a(i)
            rewards.append(mini_batch[i][2]) #Store r(i)
            new_state[i]=mini_batch[i][3] #Allocate s'(i) for the target network array from iteration i in the batch
            dones.append(mini_batch[i][4]) 
            #update y based on the last rewards and done variable
            
        # print(actual_state)
        # print(self.action_space)
        q_val = self.model.predict(actual_state) #Generate target values for training the inner loop network using the network model
        q_val_next_state = self.model.predict(new_state) #for DDQN
        target_val = self.target_model.predict(new_state) #Generate the target values for training the outer loop target network
        
        # print("target_val: ",target_val)
        for i in range(self.batch_size):
            # print("Reward is: ",rewards[i])
            if dones[i] == True:
                q_val[i][actions[i]]= rewards[i]
            else:
                # print("amax: ",np.amax(target_val[i]))
                # print("amax: ",np.argmax(q_val_next_state[i]),type(np.argmax(q_val_next_state[i])))
                # q_val[i][actions[i]]= rewards[i] + self.discount_factor*np.amax(target_val[i])
                q_val[i][actions[i]] = rewards[i] + self.discount_factor*target_val[i][np.argmax(q_val_next_state[i])] #Double DQN

            # print("q_val_i :",q_val[i])
            # y[i] = q_val[i]
            # y[i][actions[i]] = new_q
        # print("y=", y )
        # print("Arrivo a trainare")
        self.model.fit(actual_state, q_val, batch_size=self.batch_size, epochs =1, verbose=0, callbacks=[tensorboard_callback],)
     



if __name__ == '__main__':
    ENV_NAME ='CartPole-v0'
    env = gym.make(ENV_NAME)

    agent = DQNAgent(env.observation_space, env.action_space)
    path = os.path.expanduser('~/saved_models_and_log/')
    if not os.path.exists(path):
        os.makedirs(path)
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M")
    model_path = path+ENV_NAME+dt_string
    log_path = path+"/logs/"+ENV_NAME+dt_string
    print(model_path)
    #list of scores for each episode
    scores, episodes = [], [] 

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_path)

    def handler(signum, frame):
        print('Sigint detected, closing environment: ', signum)
        env.close()
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)
    
    start_time = time.time()
    test_states = np.zeros((agent.test_state_no, agent.obs_space))

    max_q = np.zeros((EPISODES, agent.test_state_no))
    max_q_mean = np.zeros((EPISODES,1))
    total_reward = 0

    done=True
    for i in range(agent.test_state_no):
        if done:
            done = False
            state = env.reset()
            state = np.reshape(state, [1, agent.obs_space])
            test_states[i] = state
        else:
            action = random.randrange(agent.action_space)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, agent.obs_space])
            test_states[i] = state
            state = next_state

    for e in range(EPISODES):
        done = False
        cumulated_reward=0
        state = env.reset()
        # print("In DQN State: ", type(state), state.shape, state)

        tmp = agent.model.predict(test_states)
        max_q[e][:] = np.max(tmp, axis=1)
        max_q_mean[e] = np.mean(max_q[e][:])


        while not done:
            env.render()
            # print("get_action")
            action = agent.get_action(state,e)
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
                total_reward += cumulated_reward
                if (e%100 == 0):
                    agent.model.save(model_path)
                    print("Model saved in path: ", model_path)
                print("episode: {}/{} score: {} avg_score: {:.2f}, q_value: {} memory length: {}".format(e, EPISODES, cumulated_reward, total_reward/(e+1), max_q_mean[e] , len(agent.buffer)))

    print("Training finito")