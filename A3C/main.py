import tensorflow as tf
import numpy as np
from A3C.model import ACNetwork
from A3C.worker import Worker
import gym
import gym_square
import os
import multiprocessing
import threading
from osim.env import RunEnv


class A3CAgent:
    def __init__(self, max_episode_length, gamma, state_space_size, action_space_size, load_model, model_path,
                 learning_rate, environment, state_space_dims):
        self.environment = environment
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.load_model = load_model
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        self.gamma = gamma
        self.max_episode_length = max_episode_length

        with tf.device("/cpu:0"):
            self.global_episodes = tf.Variable(0, dtype=tf.int32, name="global_episodes", trainable=False)
            self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.master_network = ACNetwork(action_space_size=action_space_size, state_space_size=state_space_size,
                                            keep_prob=0.6, state_dims=state_space_dims, trainer=self.trainer)
            num_workers = multiprocessing.cpu_count()
            self.workers = map(range(num_workers), lambda i: Worker(trainer=self.trainer,
                                                                    state_dims=state_space_dims,
                                                                    keep_prob=0.6,
                                                                    environment=environment,
                                                                    action_space_size=action_space_size,
                                                                    state_space_size=state_space_size,
                                                                    global_episodes=self.global_episodes,
                                                                    model_path=model_path, name=i))
            self.saver = tf.train.Saver(max_to_keep=5)

    def train(self):
        with tf.Session() as session:
            coord = tf.train.Coordinator()
            if self.load_model:
                print("Loading pre saved model")
                checkpoint = tf.train.get_checkpoint_state(self.model_path)
                self.saver.restore(session, checkpoint.model_checkpoint_path)
            else:
                session.run(tf.global_variables_initializer())

        worker_threads = []
        for w in self.workers:
            # create a thread for each worker
            thread = threading.Thread(target=lambda: w.work(coord=coord, gamma=self.gamma,
                                                            max_episode_length=self.max_episode_length,
                                                            session=session, saver=self.saver))
            thread.start()
            worker_threads.append(thread)
        coord.join(worker_threads)


def main():
    """
    Definitions for hyper parameters to be used in the RL problem
    """
    environment = RunEnv(visualize=True)
    max_episode_length = 300
    gamma = .99  # discount rate for advantage estimation and reward discounting
    s_size = len(environment.observation_space)
    a_size = len(environment.action_space)
    load_model = False
    model_path = './model'
    learning_rate = 0.01

    agent = A3CAgent(gamma=gamma, max_episode_length=max_episode_length, action_space_size=a_size,
                     state_space_size=s_size, environment=environment, load_model=load_model,
                     state_space_dims=[10, 10], model_path=model_path, learning_rate=learning_rate)
    agent.train()


if __name__ == "__main__":
    main()
