from collections import deque
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import random
import os
import gym
import operator
import functools


class TrainingConfig:
    def __init__(self, batch_size, update_frequency, discount_factor, startE, endE, annealing_steps,
                 num_episodes, pre_train_steps, max_episode_length, load_model, path, tau, state_dim):
        self.state_dim = state_dim
        self.annealing_steps = annealing_steps
        self.tau = tau
        self.endE = endE
        self.path = path
        self.startE = startE
        self.discount_factor = discount_factor
        self.load_model = load_model
        self.max_episode_length = max_episode_length
        self.pre_train_steps = pre_train_steps
        self.num_episodes = num_episodes
        self.update_frequency = update_frequency
        self.batch_size = batch_size


class ExperienceReplayBuffer:
    def __init__(self, buffer_size=50000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_experience(self, sample_size):
        return np.reshape(np.array(random.sample(self.buffer, sample_size)), [sample_size, 5])


def save_model(session, path, i):
    pass


class DeepQNetwork:

    class QNetwork:
        def _create_model(self):
            self.flattened_dim = functools.reduce(operator.mul, self.state_dim)
            self.flattened_image = tf.placeholder(shape=[None, self.flattened_dim], dtype=tf.float32)
            self.model = tf.reshape(self.flattened_image, shape=[-1, 84, 84, 3])
            self.model = slim.stack(self.model, slim.conv2d,
                                    stack_args=[(32, [8, 8], [4, 4]), (64, [4, 4], [2, 2]), (64, [3, 3], [1, 1]),
                                                (32, [7, 7], [1, 1])], scope="model")
            self.model = tf.reshape(slim.max_pool2d(self.model, [2, 2]), shape=[-1])
            self.model = slim.fully_connected(self.model, activation_fn=tf.nn.sigmoid, num_outputs=self.environment.actions)
            self.predict = tf.argmax(self.model, 1)

        def _define_loss(self):
            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_one_hot = tf.one_hot(self.actions, self.environment.actions, dtype=tf.float32)
            self.Q = tf.reduce_sum(tf.multiply(self.model, self.actions_one_hot), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
            self.update_step = self.optimisation_method.minimise(self.loss)

        def __init__(self, environment, optimisation_method, state_dim):
            self.state_dim = state_dim
            self.optimisation_method = optimisation_method
            self.environment = environment
            # create the convolutional neural network model which represents our estimated Q-function
            self._create_model()
            # define the loss function and set the optimisation method
            self._define_loss()

    @staticmethod
    def _update_target_network(tau, tf_vars):
        total_vars = len(tf_vars)
        op_holder = []
        for idx, var in enumerate(tf_vars):
            op_holder.append(tf_vars[idx + total_vars//2].assign((var.value()*tau) + (1 - tau)*tf_vars[idx + total_vars//2].value()))
        return op_holder

    @staticmethod
    def _run_update_target(session, ops):
        session.run(ops)

    def __init__(self, environment, optimisation_method, state_dim):
        self.environment = environment
        self.main_Q_network = self.QNetwork(environment=environment,
                                            optimisation_method=optimisation_method,
                                            state_dim=state_dim)
        self.target_Q_network = self.QNetwork(environment=environment,
                                              optimisation_method=optimisation_method,
                                              state_dim=state_dim)

    def train(self, config: TrainingConfig):
        init = tf.global_variables_initializer()
        model_saver = tf.train.Saver()
        trainable_variables = tf.trainable_variables()
        target_ops = DeepQNetwork._update_target_network(tau=config.tau, tf_vars=trainable_variables)
        experience_buffer = ExperienceReplayBuffer()

        # set random action decay
        e = config.startE
        step_drop = (config.startE - config.endE)/config.annealing_steps

        total_reward_per_episode = []
        total_steps = 0

        if not os.path.exists(config.path):
            os.makedirs(config.path)

        with tf.Session() as session:
            session.run(init)
            if config.load_model:
                print("Loading model from path: " + config.path)
                checkpoint = tf.train.get_checkpoint_state(config.path)
                model_saver.restore(session, checkpoint.model_checkpoint_path)

            for i in range(config.num_episodes):
                # create a new buffer and set the state to be the beginning of the environment
                episode_buffer = ExperienceReplayBuffer()
                state = self.environment.reset()
                accumulated_reward = 0
                for j in range(config.max_episode_length):
                    # choose an action to take by the e-greedy strategy (randomly if we are in the exploratory phase)
                    if total_steps > config.pre_train_steps:
                        action = self._e_greedy_action(e=e,
                                                       session=session,
                                                       state=state)
                    else:
                        action = self._select_action(state=state, session=session)
                        
                    # take a step in the environment with the chosen action
                    new_state, reward, done = self.environment.step(action)
                    new_state = np.reshape(new_state, newshape=[self.main_Q_network.flattened_dim])
                    total_steps += 1
                    # add the experience to the episode buffer
                    episode_buffer.add_experience(np.reshape(np.array([state, action, reward, new_state, done]), [1, 5]))

                    if total_steps > config.pre_train_steps:
                        # if we have completely the pure exploratory phase
                        if e > config.endE:
                            # gradually decrease the probability parameter for selecting random actions
                            e -= step_drop

                        if total_steps % config.update_frequency == 0:
                            training_batch = experience_buffer.sample_experience(config.batch_size)
                            Q_network_action_1 = session.run(self.main_Q_network.predict,
                                                             feed_dict={self.main_Q_network.flattened_image:
                                                                        np.vstack(training_batch[:, 3])})
                            Q_network_action_2 = session.run(self.target_Q_network.predict,
                                                             feed_dict={self.target_Q_network.flattened_image:
                                                                        np.vstack(training_batch[:, 3])})
                            end_multiplier = -(training_batch[:, 4] - 1)
                            # create the double Q network which implements the DDQN architecture
                            double_Q = Q_network_action_2[range(config.batch_size), Q_network_action_1]

                            # update the target network using the double Q network
                            target_Q = training_batch[:, 2] + (config.discount_factor*double_Q*end_multiplier)

                            # run the update step using the optimisation method
                            session.run(self.main_Q_network.update_step,
                                        feed_dict={self.main_Q_network.flattened_image: np.vstack(training_batch[:, 0]),
                                                   self.main_Q_network.targetQ: target_Q,
                                                   self.main_Q_network.actions: training_batch[:, 1]})
                            DeepQNetwork._update_target_network(target_ops, session)

                    accumulated_reward += reward
                    state = new_state
                    if done:
                        break
                # update the main experience replay buffer
                experience_buffer.add_experience(episode_buffer.buffer)
                total_reward_per_episode.append(accumulated_reward)

                if i % 1000 == 0:
                    save_model(session, config.path, i)
                if len(total_reward_per_episode) % 10 == 0:
                    print(total_steps, np.mean(total_reward_per_episode[-10:]), e)
        print("Percent of successful episodes: " + str(sum(total_reward_per_episode) / config.num_episodes) + "%")

    def _select_action(self, session, state):
        return session.run(self.main_Q_network.predict,
                           feed_dict={self.main_Q_network.flattened_image: [state]})[0]

    def _e_greedy_action(self, e, session, state):
        if np.random.rand(1) < e:
            return np.random.randint(0, 4)
        else:
            return session.run(self.main_Q_network.predict,
                               feed_dict={self.main_Q_network.flattened_image: [state]})[0]


def main():
    batch_size = 32  # How many experiences to use for each training step.
    update_freq = 4  # How often to perform a training step.
    y = .99  # Discount factor on the target Q-values
    start_e = 1  # Starting chance of random action
    end_e = 0.1  # Final chance of random action
    annealing_steps = 10000  # How many steps of training to reduce startE to endE.
    num_episodes = 10000  # How many episodes of game environment to train network with.
    pre_train_steps = 10000  # How many steps of random actions before training begins.
    max_ep_length = 50  # The max allowed length of our episode.
    load_model = False  # Whether to load a saved model.
    path = "./dqn-models"  # The path to save our model to.
    tau = 0.001  # Rate to update target network toward primary network
    environment = gym.make("Breakout-v0")
    config = TrainingConfig(annealing_steps=annealing_steps, batch_size=batch_size, discount_factor=y,
                            update_frequency=update_freq, startE=start_e, endE=end_e, num_episodes=num_episodes,
                            pre_train_steps=pre_train_steps, max_episode_length=max_ep_length, load_model=load_model,
                            path=path, tau=tau, state_dim=[201, 160, 3])
    dqn = DeepQNetwork(environment=environment, optimisation_method=tf.train.AdamOptimizer(learning_rate=0.01))
    dqn.train(config=config)


if __name__ == "__main__":
    main()
