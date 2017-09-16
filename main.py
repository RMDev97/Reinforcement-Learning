from BasicRL.nbandit import SimpleNBandit
import tensorflow as tf


def main():
    bandit_problem = SimpleNBandit(
        bandits=[0.2, 0, -0.2, -5, 0.7, -10, -2, 3],
        num_episodes=2000,
        optimiser_method=tf.train.GradientDescentOptimizer(learning_rate=0.001))

    bandit_problem_2 = SimpleNBandit(
        bandits=[0.2, 0, -0.2, -5, 0.7, -10, -2, 3],
        num_episodes=2000,
        optimiser_method=tf.train.AdamOptimizer(learning_rate=0.001))

    bandit_problem.train()
    bandit_problem_2.train()

    bandit_problem.get_result()
    bandit_problem_2.get_result()

if __name__ == "__main__":
    main()
