import BasicRL.nbandit_contextual as cb
import BasicRL.nbandit as b
import tensorflow as tf
import numpy as np


def contextual_bandit_test():
    bandits = np.array([[0.2, 0, -0.0, -5], [0.1, -5, 1, 0.25], [-5, 5, 5, 5]])
    agent = cb.PolicyBasedAgent(optimiser_method=tf.train.GradientDescentOptimizer(0.001),
                                state_space_size=bandits.shape[0],
                                action_space_size=bandits.shape[1],
                                num_episodes=10000)
    cbandit = cb.ContextualNBandit(agent=agent, bandits=bandits)
    cbandit.train()
    cbandit.get_result()


def bandit_test():
    bandit_problem = b.SimpleNBandit(
        bandits=[0.2, 0, -0.2, -5, 0.7, -10, -2, 3],
        num_episodes=2000,
        optimiser_method=tf.train.GradientDescentOptimizer(learning_rate=0.001))

    bandit_problem_2 = b.SimpleNBandit(
        bandits=[0.2, 0, -0.2, -5, 0.7, -10, -2, 3],
        num_episodes=2000,
        optimiser_method=tf.train.AdamOptimizer(learning_rate=0.001))

    bandit_problem.train()
    bandit_problem_2.train()

    bandit_problem.get_result()
    bandit_problem_2.get_result()

if __name__ == "__main__":
    contextual_bandit_test()
