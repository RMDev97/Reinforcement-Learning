import A3C.model as model
import tensorflow as tf
import numpy as np


class Worker:
    def __init__(self, environment, name, state_space_size, action_space_size, trainer, model_path, global_episodes,
                 state_dims, keep_prob):
        self.global_episodes = global_episodes
        self.model_path = model_path
        self.trainer = trainer
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        self.name = "worker_%s" % name
        self.environment = environment
        self.increment = self.global_episodes.assign_add(1)

        # store metrics for episodic lengths and performance
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_%s" % name)

        self.local_ac = model.A3CNetwork(state_space_size=state_space_size, action_space_size=action_space_size,
                                         keep_prob=keep_prob, state_dims=state_dims, scope=self.name, trainer=trainer)
        self.update_local_ops = model.update_target_graph('global', self.name)