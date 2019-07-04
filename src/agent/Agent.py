import numpy as np
import tensorflow as tf
from tensorflow.python.layers.layers import Dense, MaxPooling2D, Conv2D


class Agent:
    """
    this class implements an actor critic rl approach
    """
    def __init__(self, learning_rate: float, epsilon: float):
        self.network = self._configure_network()
        self.target_network = self._configure_network()
        self._gamma = 0.99
        self.optimizer = tf.optimizers.Adam(learning_rate)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def _configure_network(self):
        network = tf.python.keras.models.Sequential([
            # add layers
        ])
        return network

    def make_decision(self, state_space: np.ndarray):
        pass

    def train_agent(self, states, actions, next_states, rewards, is_done):
        # Decorator autographs the function
        @tf.function
        def td_loss():
            qvalues, state_values = self.network(states)
            next_qvalues, next_state_values = self.target_network(next_states)
            next_state_values = next_state_values * (1 - is_done)
            probs = tf.nn.softmax(qvalues)
            logprobs = tf.nn.log_softmax(qvalues)

            logp_actions = tf.reduce_sum(logprobs * tf.one_hot(actions, 4), axis=-1)
            advantage = rewards + self._gamma * next_state_values - state_values
            entropy = -tf.reduce_sum(probs * logprobs, 1, name="entropy")
            actor_loss = - tf.reduce_mean(logp_actions * tf.stop_gradient(advantage)) - 0.001 * \
                         tf.reduce_mean(entropy)
            target_state_values = rewards + self._gamma * next_state_values
            critic_loss = tf.reduce_mean((state_values - tf.stop_gradient(target_state_values)) ** 2)
            return actor_loss + critic_loss

        with tf.GradientTape() as tape:
            loss = td_loss()

        grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        return loss

