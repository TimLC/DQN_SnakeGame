import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np

from dqn.experience import Experience
from dqn.model import Model


class Agent:
    def __init__(self, env, number_hide_layers, number_neural_by_layer, path_save, buffer_size, batch_size,
                 epsilon_decay, gamma, replace):
        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.MeanSquaredError()
        self.env = env
        self.memory = Experience(buffer_size, env.state_size.n, env.action_space.n)
        self.dqn = Model(env.state_size.n, env.action_space.n, number_hide_layers, number_neural_by_layer)
        self.target_dqn = Model(env.state_size.n, env.action_space.n, number_hide_layers, number_neural_by_layer)
        self.dqn.compile(optimizer=optimizer, loss=loss)
        self.target_dqn.compile(optimizer=optimizer, loss=loss)
        self.dqn.predict(tf.zeros([1, 11], dtype=dtypes.int32))
        self.target_dqn.predict(tf.zeros([1, 11], dtype=dtypes.int32))
        self.path_save = path_save
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.replace = replace
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.training_step = 0

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = self.env.get_random_action()
        else:
            action = [0, 0, 0]
            state = np.array([state])
            predicted_action = self.dqn.predict(state)
            action[tf.math.reduce_max(predicted_action)] = 1
        return action

    def train(self):
        if self.memory.pointer < self.batch_size:
            return

        if self.training_step % self.replace == 0:
            self.update_target_model()

        states, actions, rewards, next_states, dones = self.memory.sample_of_experiences(self.batch_size)
        action_target = self.dqn.predict(states)
        next_action_target = self.target_dqn.predict(next_states)
        q_next = tf.math.reduce_max(next_action_target, axis=1, keepdims=True).numpy()
        for index in range(len(states)):
            action_target[index, np.argmax(actions[index])] = rewards[index] + self.gamma * q_next[index] * (
                        1 - dones[index])
        self.dqn.train_on_batch(states, action_target)
        self.update_epsilon()
        self.training_step += 1

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.add_one_experience(state, action, reward, next_state, done)

    def update_target_model(self):
        self.target_dqn.set_weights(self.dqn.get_weights())

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon

    def save_model(self, id):
        self.dqn.save(self.path_save + '/model_' + str(id), save_format='tf')
