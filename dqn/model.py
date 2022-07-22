import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self, state_size, action_size, number_hide_layers, number_neural_by_layer):
        super().__init__()
        self.number_hide_layers = number_hide_layers
        self.dense_start = tf.keras.layers.Dense(state_size, activation='relu')
        self.list_of_dense_hide = [tf.keras.layers.Dense(number_neural_by_layer, activation='relu') for _ in range(number_hide_layers)]
        self.dense_end = tf.keras.layers.Dense(action_size)
        self.number_neural_by_layer = number_neural_by_layer

    def call(self, input_data):
        x = self.dense_start(input_data)
        for dense_hide in self.list_of_dense_hide:
            x = dense_hide(x)
        return self.dense_end(x)
