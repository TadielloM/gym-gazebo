import numpy as np
import ray
import gym
from gym import wrappers
import gym_gazebo
import six
from ray import tune
from ray.tune.registry import register_env
# from gym_gazebo.envs.gazebo_env import GazeboEnv
# from gym_gazebo.envs.slam_exploration import GazeboSlamExplorationEnv




@ray.remote
class Runner(object):
    """Actor object to start running simulation on workers.
        Gradient computation is also executed on this object."""
    def __init__(self, env_name, actor_id):
        # starts simulation environment, policy, and thread.
        # Thread will continuously interact with the simulation environment
        self.env = env_name
        self.id = actor_id
        self.policy = LSTMPolicy()
        self.runner = RunnerThread(env, self.policy, 20)
        self.start()

    def start(self):
        # starts the simulation thread
        print("Runner n", self.id," starting") 
        self.runner.start_runner()

    def pull_batch_from_queue(self):
        # Implementation details removed - gets partial rollout from queue
        return rollout

    def compute_gradient(self, params):
        self.policy.set_weights(params)
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
        gradient = self.policy.compute_gradients(batch)
        info = {"id": self.id,
                "size": len(batch.a)}
        return gradient, info

ray.init()
EXPERIMENT_NAME = "GazeboSlamExploration-v0"

def env_creator(env_name):
    if env_name == 'GazeboSlamExploration-v0':
        from gym_gazebo.envs.slam_exploration import GazeboSlamExplorationEnv as env
    else:
        raise NotImplementedError
    return env
env = env_creator(EXPERIMENT_NAME)
tune.register_env(EXPERIMENT_NAME, lambda config : env(config))
# config = {
# "env": GazeboSlamExplorationEnv,  # or "corridor" if registered above
# "env_config": {
#     "corridor_length": 5,
# }}
# tune.run()
worker = Runner.remote(env,1)
while True:
    pass

# # We use global shared counter T_GLOBAL, and TMAX constant
# global TMAX, T_GLOBAL
# # Unpack graph ops
# s, a, R, minimize, p_network, v_network = graph_ops
    
# # outdir = '/tmp/gazebo_gym_experiments'
# # env = gym.wrappers.Monitor(env, outdir, force=True)

# # Unpack tensorboard summary stuff
# r_summary_placeholder, update_ep_reward, val_summary_placeholder, update_ep_val, summary_op = summary_ops

# # time.sleep(5*num)

# # Set up per-episode counters
# ep_reward = 0
# ep_avg_v = 0
# v_steps = 0
# ep_t = 0

# probs_summary_t = 0

# #reset the environement and return first observation
# state_t = env.reset()
# terminal = False #If it is the terminal state (done)

# while T_GLOBAL < TMAX:
#     s_batch = []
#     past_rewards = []
#     a_batch = []

#     t_local = 0
#     t_start = t_local

#     while not (terminal or ((t_local - t_start)  == t_max)):
#         # Perform action a_t according to policy pi(a_t | state_t)
#         probs = session.run(p_network, feed_dict={s: [state_t]})[0]
#         action_index = sample_policy_action(NUM_ACTIONS, probs)
#         a_t = np.zeros([NUM_ACTIONS])
#         a_t[action_index] = 1

#         if probs_summary_t % 20 == 0:
#             print "P, ", np.max(probs), "V ", session.run(v_network, feed_dict={s: [state_t]})[0][0]
            
#         s_batch.append(state_t)
#         a_batch.append(a_t)
#         #state_t1 is the new observation
#         # print "Thread num: ",num, " action: ", action_index 
#         state_t1, reward_t, terminal, info = env.step(action_index)
#         ep_reward += reward_t

#         #TODO: limit the reward to a certain value

#         past_rewards.append(reward_t)

#         t_local += 1
#         T_GLOBAL += 1
#         ep_t += 1
#         probs_summary_t += 1
            
#         state_t = state_t1

#     if terminal:
#         R_t = 0
#     else:
#         R_t = session.run(v_network, feed_dict={s: [state_t]})[0][0] # Bootstrap from last state

#     R_batch = np.zeros(t_local)
#     for i in reversed(range(t_start, t_local)):
#         R_t = past_rewards[i] + GAMMA * R_t
#         R_batch[i] = R_t

#     session.run(minimize, feed_dict={R : R_batch,
#                                     a : a_batch,
#                                     s : s_batch})
        
#     # Save progress every 100 iterations
#     if T_GLOBAL % CHECKPOINT_INTERVAL == 0:
#         saver.save(session, CHECKPOINT_SAVE_PATH, global_step = T_GLOBAL)

#     if terminal:
#         # Episode ended, collect stats and reset game
#         session.run(update_ep_reward, feed_dict={r_summary_placeholder: ep_reward})
#         print "THREAD:", num, "/ TIME", T_GLOBAL, "/ REWARD", ep_reward
#         state_t = env.reset()
#         terminal = False
#         # Reset per-episode counters
#         ep_reward = 0
#         ep_t = 0