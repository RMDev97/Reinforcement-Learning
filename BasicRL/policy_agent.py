import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import gym.wrappers as wrappers


def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add*gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards


def reset_grad_buffer(grad_buffer):
    for ix, grad in enumerate(grad_buffer):
        grad_buffer[ix] = grad * 0


class PolicyAgent:
    def __init__(self, optimisation_method, state_space_size, hidden_layer_sizes, action_space_size, environment):
        # construct a feed forward neural network given an input size, output size and the sizes of the hidden layers
        self.environment = environment
        self._create_model(state_space_size, hidden_layer_sizes, action_space_size)

        self.selected_action = tf.argmax(self.output_layer, 1)
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output_layer)[0]) * tf.shape(self.output_layer)[1] + self.action
        self.policy = tf.gather(tf.reshape(self.output_layer, [-1]), self.indexes)

        # define the loss function which will be used to optimise the parameters of the network
        self.loss = -tf.reduce_mean(tf.log(self.policy)*self.reward)

        # compute the gradients for all the variables to be trained by backprop and the optimisation algo
        self.trainable_variables = tf.trainable_variables()
        self.gradients_ph = []
        for idx, _ in enumerate(self.trainable_variables):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+"_holder")
            self.gradients_ph.append(placeholder)

        self.gradients = tf.gradients(self.loss, self.trainable_variables)

        self.update_batch_step = optimisation_method.apply_gradients(zip(self.gradients_ph, self.trainable_variables))

    def _create_model(self, state_space_size, hidden_layer_sizes, action_space_size):
        self.input_layer = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)
        hidden_layer = slim.fully_connected(self.input_layer, hidden_layer_sizes[0],
                                            biases_initializer=None, activation_fn=tf.nn.relu)

        for i in range(1, len(hidden_layer_sizes)):
            hidden_layer = slim.fully_connected(hidden_layer,
                                                hidden_layer_sizes[i],
                                                biases_initializer=None,
                                                activation_fn=tf.nn.relu)

        self.output_layer = slim.fully_connected(hidden_layer,
                                                 action_space_size,
                                                 activation_fn=tf.nn.softmax,
                                                 biases_initializer=None)

    def _select_action(self, session, state):
        action_dictionary = session.run(self.output_layer, feed_dict={self.input_layer: [state]})
        action = np.argmax(np.equal(
            action_dictionary,
            np.random.choice(action_dictionary[0], p=action_dictionary[0]))
                           .astype(int))
        return action

    def train(self, num_episodes, max_episode, update_frequency):
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            total_reward = []

            grad_buffer = session.run(tf.trainable_variables())
            reset_grad_buffer(grad_buffer)

            for i in range(num_episodes):
                state = self.environment.reset()
                running_reward = 0
                episode_history = []
                for j in range(max_episode):
                    # select an action probabilistically in order to build up experience buffer
                    action = self._select_action(session, state)
                    new_state, reward, done, _ = self.environment.step(action)
                    # add to experience buffer
                    episode_history.append([state, action, reward, new_state])
                    state = new_state
                    running_reward += reward
                    # if we have reached the goal state then update the network with the experience
                    # and reset the gradients
                    if done:
                        # compute the gradients from the experience buffer
                        episode_history = np.array(episode_history)
                        episode_history[:, 2] = discount_rewards(episode_history[:, 2], 0.99)
                        gradients = session.run(self.gradients,
                                                feed_dict={self.reward: episode_history[:, 2],
                                                           self.action: episode_history[:, 1],
                                                           self.input_layer: np.vstack(episode_history[:, 0])})
                        for idx, grad in enumerate(gradients):
                            grad_buffer[idx] += grad

                        if i % update_frequency == 0 and i != 0:
                            # run the optimiser update step for the computed gradients thus far and reset them
                            session.run(self.update_batch_step, feed_dict=dict(zip(self.gradients_ph, grad_buffer)))
                            reset_grad_buffer(grad_buffer)

                        total_reward.append(running_reward)
                        break

                if i % 100 == 0:
                    print("Current mean of running total reward at iteration %d is %s" %
                          (i, np.mean(total_reward[-100:])))


def main():
    environment = gym.make('CartPole-v0')
    environment = wrappers.Monitor(environment, './experiments/cartpole-experiment-1',
                                   video_callable=False, write_upon_reset=True, force=True)
    agent = PolicyAgent(state_space_size=4,
                        hidden_layer_sizes=[8],
                        action_space_size=2,
                        environment=environment,
                        optimisation_method=tf.train.AdamOptimizer(0.01))
    agent.train(max_episode=999, num_episodes=100, update_frequency=5)

if __name__ == "__main__":
    main()
