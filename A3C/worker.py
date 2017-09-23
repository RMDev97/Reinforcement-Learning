from collections import deque

import A3C.model as model
import tensorflow as tf
import numpy as np


class ExperienceReplayBuffer:
    def __init__(self, buffer_size=50000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add_experience(self, experience):
        self.buffer.append(experience)


def discount(x, gamma):
    discounts = map(range(len(x)), lambda n: gamma ** n)
    return np.multiply(x, discounts)


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

        self.local_ac = model.LocalA3CNetwork(model.A3CNetwork(state_space_size=state_space_size,
                                              action_space_size=action_space_size, keep_prob=keep_prob,
                                              state_dims=state_dims, scope=self.name, trainer=trainer))
        self.update_local_ops = model.update_target_graph('global', self.name)

    def train(self, rollout, session, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = discount(rewards + gamma*self.value_plus[:-1] - self.value_plus[:-1], gamma)

        # update the global network with the gradients derived from optimising the local network
        feed_dict = {
            self.local_ac.target_v: discounted_rewards,
            self.local_ac.network.inputs: np.vstack(observations),
            self.local_ac.actions: actions,
            self.local_ac.advantages: advantages,
            self.local_ac.network.state_in[0]: self.batch_rnn_state[0],
            self.local_ac.network.state_in[1]: self.batch_rnn_state[1]
        }
        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = session.run([self.local_ac.value_loss,
                                                                        self.local_ac.policy_loss,
                                                                        self.local_ac.entropy,
                                                                        self.local_ac.grad_norms,
                                                                        self.local_ac.var_norms,
                                                                        self.local_ac.network.state_out,
                                                                        self.local_ac.apply_grads
                                                                        ],
                                                                       feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, session: tf.Session, coord, saver):
        episode_count = session.run(self.global_episodes)
        total_steps = 0
        with session.as_default(), session.graph.as_default():
            while not coord.should_stop():
                session.run(self.update_local_ops)
                episode_buffer = ExperienceReplayBuffer(30)
                episode_reward = 0

                state = self.environment.reset()
                rnn_state = self.local_ac.network.state_init

                # run each episode of the training
                for episode_step_count in range(0, max_episode_length):
                    action_predict, value_predict, rnn_state = session.run([self.local_ac.network.actor,
                                                                            self.local_ac.network.critic,
                                                                            self.local_ac.network.state_out],
                                                                           feed_dict={
                                                                               self.local_ac.network.state_in: rnn_state,
                                                                               self.local_ac.network.inputs: [state]
                                                                           })
                    action = self.select_action(action_predict)
                    new_state, reward, done, _ = self.environment.step(action)

                    episode_buffer.add_experience([state, action, reward, new_state, done, value_predict[0, 0]])
                    episode_reward += reward
                    state = new_state
                    total_steps += 1

                    if total_steps % 30 == 0:
                        # estimate the final return based on the critic model
                        estimated_value = session.run(self.local_ac.network.critic, feed_dict={
                            self.local_ac.network.inputs: [state],
                            self.local_ac.network.state_in: rnn_state
                        })[0, 0]
                        self.train(episode_buffer.buffer, session, gamma, estimated_value)
                    if done:
                        break
                self.episode_rewards.append(episode_reward)

                # at the end of the episode, train the network on whatever remains in the buffer
                if len(episode_buffer.buffer) != 0:
                    self.train(episode_buffer.buffer, session, gamma, 0.0)

                # save the current model every 200 episodes or so
                if episode_count % 250 == 0 and self.name == 'worker_0':
                    saver.save(session, '%s/model-%s.ckpt' % (self.model_path, episode_count))
                    print("Saved Model")

                if episode_count % 100 and episode_count != 0:
                    print("mean reward at episode %s: %s" % (episode_count, np.mean(self.episode_rewards[-100:0])))

                if self.name == "worker_0":
                    session.run(self.increment)

                episode_count += 1
