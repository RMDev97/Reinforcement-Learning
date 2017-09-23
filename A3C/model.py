"""
This file includes implementations of various models which are used in the A3C RL agent architecture.

We include in this file implementations of the main A3C network that will be used to combine all the estimates from
each of the worker agents, branches for both the policy estimator models and the value estimator models (respectively
the actor and critic models)

the global network itself consists of a few initial convolutional neural network layers to handle spatial dependencies
of the inputs followed by LSTM layers to handle the temporal dependencies
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


class ACNetwork:

    def create_kernel(self, x):
        return [x for _ in range(len(self.state_dims))]

    def __init__(self, state_space_size, action_space_size, trainer, state_dims, keep_prob, scope='global'):
        # build the input and visual encoding layers through use of convolutional neural network layers
        self.state_dims = state_dims
        self.trainer = trainer
        self.scope = scope
        self.keep_prob = keep_prob
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size

        self.inputs = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)
        self.image_input_layer = tf.reshape(self.inputs, shape=[-1, state_dims, 1])
        self.output_layer = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.image_input_layer, num_outputs=16,
                                        kernel_size=self.create_kernel(8), stride=self.create_kernel(4),
                                        padding="VALID")
        self.output_layer = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.output_layer, num_outputs=32,
                                        kernel_size=self.create_kernel(4), stride=self.create_kernel(2),
                                        padding="VALID")
        self.output_layer = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.output_layer, num_outputs=32,
                                        kernel_size=self.create_kernel(4), stride=self.create_kernel(2),
                                        padding="VALID")
        self.output_layer = slim.dropout(inputs=self.output_layer, keep_prob=keep_prob)
        self.output_layer = slim.fully_connected(slim.flatten(self.output_layer), 256, activation_fn=tf.nn.elu)

        # define the recurrent LSTM layers to learn temporal dependencies
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        self.state_in = (c_in, h_in)
        rnn_in = tf.expand_dims(self.output_layer, [0])
        step_size = tf.shape(self.image_input_layer)[:1]
        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        rnn_output_layer = tf.reshape(lstm_outputs, [-1, 256])

        # define the output layers for the policy (actor) and value (critic)
        self.actor = slim.fully_connected(rnn_output_layer, num_outputs=self.action_space_size,
                                          activation_fn=tf.nn.softmax,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1))

        self.critic = slim.fully_connected(rnn_output_layer, num_outputs=1,
                                           weights_initializer=tf.truncated_normal_initializer(stddev=0.1))


class LocalA3CNetwork:
    def __init__(self, network: ACNetwork):
        self.network = network
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_one_hot = tf.one_hot(self.actions, network.action_space_size, dtype=tf.float32)
        self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
        self.responsible_outputs = tf.reduce_sum(network.actor * self.actions_one_hot, [1])

        # define the loss functions to be used to train the worker network
        self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(network.critic, [-1])))
        self.entropy = tf.reduce_sum(network.actor * tf.log(network.actor))
        self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
        self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy

        # compute gradients of losses to provide to trainer
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, network.scope)
        self.gradients = tf.gradients(self.loss, local_vars)
        self.var_norms = tf.global_norm(local_vars)
        grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

        # apply the gradients to the global network
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.apply_grads = network.trainer.apply_gradients(zip(grads, global_vars))
