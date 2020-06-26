#!/usr/bin/env python

#part of the code in this file is based on the code provided by Corey Linch at https://github.com/coreylynch/async-rl

import gym
from gym import wrappers
# from gym.wrappers.monitoring import monitoring
import gym_gazebo
import time
import numpy as np
import time
import pandas
import sys
import threading
import tensorflow as tf
from keras import backend as K

from a3c import build_policy_and_value_networks
import liveplot
import ray

#params
EXPERIMENT_NAME = "GazeboSlamExploration-v0"
SAVING_PATH = "/tmp/gazebo_gym_experiments"
NUM_STEPS = 1000
NUM_AGENTS = 2
NUM_EPISODES = 100
NUM_ACTIONS = 26
EPSILON_DISCOUNT = 0.9988
SUMMARY_INTERVAL= 5
CHECKPOINT_INTERVAL = 100
CHECKPOINT_SAVE_PATH = "/tmp/"+EXPERIMENT_NAME+".ckpt"

AGENT_HISTORY_LENGTH = 4
RESIZED_X = 200
RESIZED_Y = 200
RESIZED_Z = 200
RESIZED_DATA = 2 #is voxel full is voxel empty

#Shared global parameters
T_GLOBAL = 0
TMAX = 800000000
t_max = 32

# Optimization Params
LEARNING_RATE = 0.00001

# DQN Params
GAMMA = 0.99

def render():
    render_skip = 0 #Skip first X episodes.
    render_interval = 50 #Show render Every Y episodes.
    render_episodes = 10 #Show Z episodes every rendering.

    if (x%render_interval == 0) and (x != 0) and (x > render_skip):
        env.render()
    elif ((x-render_episodes)%render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
        env.render(close=True)

def sample_policy_action(num_actions, probs):
    """
    Sample an action from an action probability distribution output by
    the policy network.
    """
    # Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
    probs = probs - np.finfo(np.float32).epsneg

    histogram = np.random.multinomial(1, probs)
    action_index = int(np.nonzero(histogram)[0])
    return action_index

def actor_learner_thread(num, env, session, graph_ops, summary_ops, saver):
    # We use global shared counter T_GLOBAL, and TMAX constant
    global TMAX, T_GLOBAL
    # Unpack graph ops
    s, a, R, minimize, p_network, v_network = graph_ops
    
    # outdir = '/tmp/gazebo_gym_experiments'
    # env = gym.wrappers.Monitor(env, outdir, force=True)

    # Unpack tensorboard summary stuff
    r_summary_placeholder, update_ep_reward, val_summary_placeholder, update_ep_val, summary_op = summary_ops

    # time.sleep(5*num)

    # Set up per-episode counters
    ep_reward = 0
    ep_avg_v = 0
    v_steps = 0
    ep_t = 0

    probs_summary_t = 0

    #reset the environement and return first observation
    state_t = env.reset()
    terminal = False #If it is the terminal state (done)

    while T_GLOBAL < TMAX:
        s_batch = []
        past_rewards = []
        a_batch = []

        t_local = 0
        t_start = t_local

        while not (terminal or ((t_local - t_start)  == t_max)):
            # Perform action a_t according to policy pi(a_t | state_t)
            probs = session.run(p_network, feed_dict={s: [state_t]})[0]
            action_index = sample_policy_action(NUM_ACTIONS, probs)
            a_t = np.zeros([NUM_ACTIONS])
            a_t[action_index] = 1

            if probs_summary_t % 20 == 0:
                print "P, ", np.max(probs), "V ", session.run(v_network, feed_dict={s: [state_t]})[0][0]
            
            s_batch.append(state_t)
            a_batch.append(a_t)
            #state_t1 is the new observation
            # print "Thread num: ",num, " action: ", action_index 
            state_t1, reward_t, terminal, info = env.step(action_index)
            ep_reward += reward_t

            #TODO: limit the reward to a certain value

            past_rewards.append(reward_t)

            t_local += 1
            T_GLOBAL += 1
            ep_t += 1
            probs_summary_t += 1
            
            state_t = state_t1

        if terminal:
            R_t = 0
        else:
            R_t = session.run(v_network, feed_dict={s: [state_t]})[0][0] # Bootstrap from last state

        R_batch = np.zeros(t_local)
        for i in reversed(range(t_start, t_local)):
            R_t = past_rewards[i] + GAMMA * R_t
            R_batch[i] = R_t

        session.run(minimize, feed_dict={R : R_batch,
                                        a : a_batch,
                                        s : s_batch})
        
        # Save progress every 100 iterations
        if T_GLOBAL % CHECKPOINT_INTERVAL == 0:
            saver.save(session, CHECKPOINT_SAVE_PATH, global_step = T_GLOBAL)

        if terminal:
            # Episode ended, collect stats and reset game
            session.run(update_ep_reward, feed_dict={r_summary_placeholder: ep_reward})
            print "THREAD:", num, "/ TIME", T_GLOBAL, "/ REWARD", ep_reward
            state_t = env.reset()
            terminal = False
            # Reset per-episode counters
            ep_reward = 0
            ep_t = 0

def build_graph():
    # Create shared global policy and value networks
    s, p_network, v_network, p_params, v_params = build_policy_and_value_networks(num_actions=NUM_ACTIONS, resized_x=RESIZED_X, resized_y=RESIZED_Y,resized_z=RESIZED_Z,resized_data=RESIZED_DATA)

    # Shared global optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE)

    # Op for applying remote gradients
    R_t = tf.compat.v1.placeholder("float", [None])
    action_t = tf.compat.v1.placeholder("float", [None, NUM_ACTIONS])
    log_prob = tf.math.log(tf.reduce_sum(p_network * action_t, reduction_indices=1))
    p_loss = -log_prob * (R_t - v_network)
    v_loss = tf.reduce_mean(tf.square(R_t - v_network))

    total_loss = p_loss + (0.5 * v_loss)

    minimize = optimizer.minimize(total_loss)
    return s, action_t, R_t, minimize, p_network, v_network

# Set up some episode summary ops to visualize on tensorboard.
def setup_summaries():
    episode_reward = tf.Variable(0.)
    tf.compat.v1.summary.scalar("Episode_Reward", episode_reward)
    r_summary_placeholder = tf.compat.v1.placeholder("float")
    update_ep_reward = episode_reward.assign(r_summary_placeholder)
    ep_avg_v = tf.Variable(0.)
    tf.compat.v1.summary.scalar("Episode_Value", ep_avg_v)
    val_summary_placeholder = tf.compat.v1.placeholder("float")
    update_ep_val = ep_avg_v.assign(val_summary_placeholder)
    summary_op = tf.compat.v1.summary.merge_all()
    return r_summary_placeholder, update_ep_reward, val_summary_placeholder, update_ep_val, summary_op

def train(session, graph_ops, saver):
    # Set up game environments (one per thread)
    envs = []

    for i in range(NUM_AGENTS):
        envs.append(gym.make(EXPERIMENT_NAME))
        outdir = SAVING_PATH+str(i)
        envs[i] = gym.wrappers.Monitor(envs[i], outdir, force=True)
        time.sleep(5)
    
    # outdir = SAVING_PATH
    # env = gym.wrappers.Monitor(env, outdir, force=True)
    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables
    session.run(tf.compat.v1.global_variables_initializer())
    writer = tf.compat.v1.summary.FileWriter(SAVING_PATH, session.graph)

    # Start NUM_AGENTS training threads
    actor_learner_threads = [threading.Thread(target=actor_learner_thread, args=(thread_id, envs[thread_id], session, graph_ops, summary_ops, saver)) for thread_id in range(NUM_AGENTS)]
    for t in actor_learner_threads:
        t.start()

    # Show the agents training and write summary statistics
    last_summary_time = 0
    while True:
        now = time.time()
        if now - last_summary_time > SUMMARY_INTERVAL:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T_GLOBAL))
            last_summary_time = now
    for t in actor_learner_threads:
        t.join()

    for i in envs:
        i.close()

if __name__ == '__main__':
    g = tf.Graph()
    with g.as_default(), tf.compat.v1.Session() as session:
        K.set_session(session)
        graph_ops = build_graph()
        saver = tf.compat.v1.train.Saver()

        train(session,graph_ops,saver)

    