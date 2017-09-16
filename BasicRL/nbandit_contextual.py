import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class ContextualNBandit:
    def __init__(self, bandits, agent):
        self.agent = agent
        self.state = 0
        self.bandits = bandits
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def get_random_state(self):
        self.state = np.random.randint(0, len(self.bandits))
        return self.state

    def get_reward(self, action):
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            return 1
        else:
            return -1

    def train(self):
        self.agent.train(self)

    def get_result(self):
        for action in range(self.num_bandits):
            print("Expected action for bandit %s is: %s "
                  % (str(action + 1), str(np.argmax(self.agent.final_weights[action]) + 1)))
            print("Actual action for bandit %s is: %s " % (str(action + 1), np.argmin(self.bandits[action]) + 1))


class PolicyBasedAgent:
    def __init__(self, optimiser_method, state_space_size, action_space_size, num_episodes):
        # create a single layer feed forward neural network that takes a state and outputs an action
        self.final_weights = None
        self.num_episodes = num_episodes
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)
        one_hot_state_in = slim.one_hot_encoding(self.state_in, state_space_size)
        output = slim.fully_connected(one_hot_state_in,
                                      action_space_size,
                                      biases_initializer=None,
                                      activation_fn=tf.nn.sigmoid,
                                      weights_initializer=tf.ones_initializer())

        self.action_output = tf.reshape(output, shape=[-1])
        self.selected_action = tf.argmax(self.action_output, 0)

        self.reward = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action = tf.placeholder(shape=[1], dtype=tf.int32)
        self.action_weight = tf.slice(self.action_output, self.action, [1])

        self.loss = -(tf.log(self.action_weight)*self.reward)
        self.optimiser_step = optimiser_method.minimize(self.loss)

    def train(self, contextual_bandit: ContextualNBandit):
        # the weights assigned to each action
        weights = tf.trainable_variables()[0]
        # set the initial reward array to zero
        total_reward = np.zeros([contextual_bandit.num_bandits, contextual_bandit.num_actions])
        e = 0.1
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            for i in range(self.num_episodes):
                state = contextual_bandit.get_random_state()
                # choose an action based on e-greedy action selection
                if np.random.randn(1) < e:
                    action = np.random.randint(0, contextual_bandit.num_actions)
                else:
                    action = session.run(self.selected_action, feed_dict={self.state_in: [state]})

                # get the reward for performing the action in the given state
                reward = contextual_bandit.get_reward(action)
                _, w = session.run([self.optimiser_step, weights],
                                   feed_dict={self.reward: [reward], self.action: [action], self.state_in: [state]})

                self.final_weights = w
                # update the rewards
                total_reward[state, action] += reward
                if i % 500 == 0:
                    print("Mean reward for each of the %s bandits: %s" %
                          (str(contextual_bandit.num_bandits), str(np.mean(total_reward, axis=1))))
