import ray
from ray.rllib import agents
from gym_gazebo.envs.slam_exploration import GazeboSlamExplorationEnv
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from ray.rllib.utils import try_import_tf
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

# from keras.layers import Conv3D, Flatten, Dense, Input
# from keras.models import Model
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
# from ray.rllib.utils.test_utils import check_learning_achieved

import sys, signal
import numpy as np
#tf = try_import_tf()
#print(tf) 

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

class custom_3DCNN(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(custom_3DCNN, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        # print("OBS_SHAPE= ",obs_space.shape)
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        shared = tf.keras.layers.Conv3D(name="conv1", filters=16, kernel_size=(20,20,20), strides=(20,20,20), activation='relu', padding='same')(self.inputs)
        # inputs = Input(shape=obs_space.shape, name="observations")
        # shared = Conv3D(name="conv1", filters=16, kernel_size=(20,20,20), strides=(20,20,20), activation='relu', padding='same')(inputs)
        shared = tf.keras.layers.Conv3D(name="conv2", filters=32, kernel_size=(4,4,4), strides=(10,10,10), activation='relu', padding='same')(shared)
        shared = tf.keras.layers.Flatten()(shared)
        shared = tf.keras.layers.Dense(name="h1", units=256, activation='relu')(shared)

        action_probs = tf.keras.layers.Dense(name="p", units=num_outputs, activation='softmax')(shared)
        
        state_value = tf.keras.layers.Dense(name="v", units=1, activation='linear')(shared)

        self.base_model = tf.keras.Model(inputs=self.inputs, outputs=[action_probs,state_value])
        self.register_variables(self.base_model.variables)
        # return state, p_out, v_out, p_params, v_params
    
    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
    # def metrics(self):
    #     return {"foo": tf.Variable()}

class custom_3DCNN_Q(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(custom_3DCNN_Q, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        # print("OBS_SHAPE= ",obs_space.shape)

        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        shared = tf.keras.layers.Conv3D(name="conv1", filters=16, kernel_size=(20,20,20), strides=(20,20,20), activation='relu', padding='same')(self.inputs)
        # inputs = Input(shape=obs_space.shape, name="observations")
        # shared = Conv3D(name="conv1", filters=16, kernel_size=(20,20,20), strides=(20,20,20), activation='relu', padding='same')(inputs)
        shared = tf.keras.layers.Conv3D(name="conv2", filters=32, kernel_size=(4,4,4), strides=(10,10,10), activation='relu', padding='same')(shared)
        shared = tf.keras.layers.Flatten()(shared)
        shared = tf.keras.layers.Dense(name="h1", units=256, activation='relu')(shared)

        action_probs = tf.keras.layers.Dense(name="p", units=num_outputs, activation='linear')(shared)
            
        self.base_model = tf.keras.Model(inputs=self.inputs, outputs=action_probs)
        # print(self.base_model.variables)
        self.register_variables(self.base_model.variables)
        # return state, p_out, v_out, p_params, v_params
        
    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state

    # def metrics(self):
    #     return {"foo": tf.Variable()}

class custom_PointNet(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(custom_PointNet, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        #adam = optimizers.Adam(lr=0.001, decay=0.7)
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        shared = tf.keras.layers.Convolution1D(64,1,activation='relu') (self.inputs)
        shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.Convolution1D(128, 1, activation='relu')(shared)
        shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.Convolution1D(1024, 1, activation='relu')(shared)
        shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.MaxPooling1D(pool_size=obs_space.shape[0])(shared)
        shared = tf.keras.layers.Dense(512, activation='relu')(shared)
        shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.Dense(256, activation='relu')(shared)

        # shared = tf.keras.layers.Flatten()(shared)
        shared = tf.keras.layers.Dense(name="h1", units=256, activation='relu')(shared)

        action_probs = tf.keras.layers.Dense(name="p", units=num_outputs, activation='softmax')(shared)
        state_value = tf.keras.layers.Dense(name="v", units=1, activation='linear')(shared)

        self.base_model = tf.keras.Model(inputs=self.inputs , outputs= [action_probs,state_value])

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

class custom_PointNet_Q(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(custom_PointNet_Q, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        shared = tf.keras.layers.Convolution1D(64,1,activation='relu') (self.inputs)
        shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.Convolution1D(128, 1, activation='relu')(shared)
        shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.Convolution1D(1024, 1, activation='relu')(shared)
        shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.MaxPooling1D(pool_size=obs_space.shape[0])(shared)
        shared = tf.keras.layers.Dense(512, activation='relu')(shared)
        shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.Dense(256, activation='relu')(shared)

        # #Aggiunte ma non so se servon0
        shared = tf.keras.layers.Flatten()(shared)
        shared = tf.keras.layers.Dense(name="h1", units=256, activation='relu')(shared)

        action_probs = tf.keras.layers.Dense(name="p", units=num_outputs, activation='linear')(shared)
        self.base_model = tf.keras.Model(inputs=self.inputs , outputs=action_probs)

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state

class custom_PointNet_new_Q(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(custom_PointNet_new_Q, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        x = tnet(self.inputs, 4)
        x = conv_bn(x, 32)
        x = conv_bn(x, 32)
        x = tnet(x, 32)
        x = conv_bn(x, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 256)
        x = layers.Dropout(0.3)(x)
        x = dense_bn(x, 128)
        x = layers.Dropout(0.3)(x)
        

        action_probs = tf.keras.layers.Dense(name="p", units=num_outputs, activation='linear')(x)
        self.base_model = tf.keras.Model(inputs=self.inputs , outputs=action_probs)

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state

def env_creator(env_name):
    if env_name == 'GazeboSlamExploration-v0':
        from gym_gazebo.envs.slam_exploration import GazeboSlamExplorationEnv as env
    else:
        raise NotImplementedError
    return env

if __name__ == "__main__":
    
    #ray.init(webui_host='127.0.0.1', num_gpus=1) #for gpus
    ray.init(dashboard_host='127.0.0.1') #fuor inkonova pc
    ModelCatalog.register_custom_model("My_3DCNN",custom_3DCNN)
    ModelCatalog.register_custom_model("My_3DCNN_Q",custom_3DCNN_Q)
    ModelCatalog.register_custom_model("My_PointNet",custom_PointNet)
    ModelCatalog.register_custom_model("My_PointNet_Q",custom_PointNet_Q)
    ModelCatalog.register_custom_model("My_PointNet_new_Q",custom_PointNet_new_Q)

    EXPERIMENT_NAME = "GazeboSlamExploration-v0"
    
    env = env_creator(EXPERIMENT_NAME)
    # tune.register_env(EXPERIMENT_NAME, lambda config : env(config))
    # config =agents.a3c.DEFAULT_CONFIG.copy()
    # config =agents.ppo.DEFAULT_CONFIG.copy()
    config =agents.dqn.DEFAULT_CONFIG.copy()
    
    config["env"] = env
    config['num_workers'] = 0
    config["model"] = {
        # "custom_model": "My_3DCNN_Q" 
        "custom_model": "My_PointNet_new_Q"
        
    }
    config["timesteps_per_iteration"] = 60
    config["train_batch_size"] = 2
    config["buffer_size"]=1000
    config["learning_starts"] = 20
    # stop = {
        # "training_iteration": 1,
        # "timesteps_total":100000000,
        # "episode_reward_mean": 50000,
    # }

    # print(pretty_print(config))

    # trainer = agents.a3c.A3CTrainer(env=env_creator(EXPERIMENT_NAME),config=config)
    # trainer = agents.ppo.PPOTrainer(env=env_creator(EXPERIMENT_NAME),config=config)
    trainer = agents.dqn.DQNTrainer(env=env_creator(EXPERIMENT_NAME),config=config)
    model = trainer.get_policy().model
    print(model.variables())
    print(model.base_model.summary())

    def handler(signum, frame):
        print('Sigint detected, closing environment: ', signum)
        trainer.workers.local_worker().env.close()
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)
    is_training = True
    if is_training==True:
        for i in range(15501):
            print("TRAINING iter nun: ", str(i))
            results = trainer.train()
            print(pretty_print(results))
            if i % 10 == 0: #save every 10th training iteration
                checkpoint_path = trainer.save()
                print("The checkpoint path is:")
                print(checkpoint_path)

        # tune.run("DQN", stop=stop, config=config)
        print("FINISHHHEEEEED TRAININNG")
        # print('Mean Rewards:\t{:.1f}'.format(results['episode_reward_mean']))
    else:
        print("Start Testing")
        checkpoint_path = "/home/tadiellomatteo/ray_results/DQN_GazeboSlamExplorationEnv_2020-10-06_06-14-20fudenrau/checkpoint_221/checkpoint-221"
        trainer.restore(checkpoint_path)
        # run until episode ends
        episode_reward = 0
        done = False
        env = trainer.workers.local_worker().env #get environmentinstance from the local worker
        obs = env.reset()
        while not done:
            action = trainer.compute_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        env.close()
        print("FINISHED TEST:\n Reward: ",str(episode_reward))

    print("Turning off ray")
    ray.shutdown()
