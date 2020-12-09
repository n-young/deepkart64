import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class DK64Model(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The DK64Model class that inherits from tf.keras.Model.
        The forward pass calculates the policy for the agent given a batch of states. During training,
        Model estimates the value of each state to be used as a baseline to compare the policy's
        performance with.
        :param state_size: number of parameters that define the state. You don't necessarily have to use this,
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(DK64Model, self).__init__()
        self.state_size = state_size
        self.num_actions = num_actions

        # Define actor network parameters, critic network parameters, and optimizer
        self.learning_rate = 1e-4
        self.epsilon = 1e-5
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.hidden_size = 200

        # Actor convolutional layers.
        self.actor_conv_1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
        )
        self.actor_leaky1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.actor_conv_2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
        )
        self.actor_leaky2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.actor_conv_3 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
        )
        self.actor_leaky3 = tf.keras.layers.LeakyReLU(alpha=0.2)

        # Actor dense layers.
        self.actor_dense1 = tf.keras.layers.Dense(self.hidden_size, "relu")
        self.actor_dense2 = tf.keras.layers.Dense(self.hidden_size, "relu")
        self.actor_dense3 = tf.keras.layers.Dense(self.num_actions)

        # Critic convolutional layers.
        self.critic_conv_1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
        )
        self.critic_leaky1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.critic_conv_2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
        )
        self.critic_leaky2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.critic_conv_3 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
        )
        self.critic_leaky3 = tf.keras.layers.LeakyReLU(alpha=0.2)

        # Critic dense layers.
        self.critic_dense1 = tf.keras.layers.Dense(self.hidden_size, "relu")
        self.critic_dense2 = tf.keras.layers.Dense(1)

    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.
        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """

        # Encode image into states.
        cnn_output = self.actor_conv_1(states)
        cnn_output = self.actor_leaky1(cnn_output)
        cnn_output = self.actor_conv_2(cnn_output)
        cnn_output = self.actor_leaky2(cnn_output)
        cnn_output = self.actor_conv_3(cnn_output)
        cnn_output = self.actor_leaky3(cnn_output)

        # Reshape CNN output and pass through dense layers.
        dense_input = tf.reshape(cnn_output, (tf.shape(states)[0], -1))
        dense_output = self.actor_dense1(dense_input)
        dense_output = self.actor_dense2(dense_output)
        dense_output = self.actor_dense3(dense_output)

        ## Return softmaxed probabilities.
        probabilities = tf.nn.softmax(dense_output)
        return probabilities

    def value_function(self, states):
        """
        Value function.
        """
        # Encode image into states.
        cnn_output = self.critic_conv_1(states)
        cnn_output = self.critic_leaky1(cnn_output)
        cnn_output = self.critic_conv_2(cnn_output)
        cnn_output = self.critic_leaky2(cnn_output)
        cnn_output = self.critic_conv_3(cnn_output)
        cnn_output = self.critic_leaky3(cnn_output)

        # Reshape CNN output and pass through dense layers.
        dense_input = tf.reshape(cnn_output, (tf.shape(states)[0], -1))
        dense_output = self.critic_dense1(dense_input)
        dense_output = self.critic_dense2(dense_output)
        return dense_output

    def loss(self, states, actions, discounted_rewards):
        """
        Loss function.
        """
        # Get values, advantage, and probabilities.
        values = tf.squeeze(self.value_function(states))
        advantage = tf.math.subtract(discounted_rewards, values)
        probabilities = self.call(states)
        probabilities_per_action = tf.gather_nd(
            probabilities,
            tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1),
        )

        # Get and return actor loss and critic loss.
        actor_loss = -1 * tf.reduce_sum(
            tf.math.log(probabilities_per_action) * tf.stop_gradient(advantage)
        )
        critic_loss = tf.reduce_sum((discounted_rewards - values) ** 2)
        return actor_loss + critic_loss
