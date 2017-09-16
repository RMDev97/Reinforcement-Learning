import tensorflow as tf
import numpy as np


# rewards will be favoured for bandits of high negative values
def get_reward(bandit):
    result = np.random.randn(1)
    if result > bandit:
        # corresponds to pulling the bandit and getting reward
        return 1
    else:
        # means that we are punished for pulling the nadit in this case
        return -1


class SimpleNBandit:
    def __init__(self, bandits, optimiser_method, num_episodes):
        self.num_episodes = num_episodes
        self.num_bandits = len(bandits)
        self.bandits = bandits
        self.final_weights = np.zeros([self.num_bandits])
        print("bandits set to: %s" % str(self.bandits))

        # create the agent model via a single layer neural network
        self.weights = tf.Variable(tf.ones([self.num_bandits]))
        self.selected_action = tf.argmax(self.weights, 0)

        self.reward = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action = tf.placeholder(shape=[1], dtype=tf.int32)
        self.action_weight = tf.slice(self.weights, self.action, [1])

        # define the loss function to be the policy gradient loss
        loss = -(tf.log(self.action_weight)*self.reward)
        self.optimise_step = optimiser_method.minimize(loss)

    def train(self):
        total_reward = np.zeros([self.num_bandits])
        e = 0.1
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            for i in range(self.num_episodes):
                # choose an action with the e-greedy action selection strategy
                if np.random.randn(1) < e:
                    action = np.random.randint(self.num_bandits)
                else:
                    action = session.run(self.selected_action)

                # get the reward from having selected the bandit to pull
                reward = get_reward(bandit=self.bandits[action])
                _, _, w = session.run([self.optimise_step, self.action_weight, self.weights],
                                      feed_dict={self.reward: [reward], self.action: [action]})
                self.final_weights = w
                # update the rewards with that received from performing the action
                total_reward[action] += reward

                if i % 50 == 0:
                    print("Running reward for the %s bandits: %s" % (str(self.num_bandits), str(total_reward)))

    def get_result(self):
        print("Agent predicted that bandit %s is the most optimal selection" % str(np.argmax(self.final_weights) + 1))
        print("Correct bandit to choose was %s." % str(np.argmax(-np.array(self.bandits))+1))

