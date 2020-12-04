import os
import sys
sys.path.insert(0, '../gym_mupen64plus/envs/MarioKart64')
import gym, gym_mupen64plus
import numpy as np
import tensorflow as tf
from observe import observe
from discrete_envs import DiscreteActions
from reinforce_with_baseline import DK64Model
from compress import compress

def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each timestep in an episode, and
    returns a list of the discounted rewards for each timestep.
    Refer to the slides to see how this is done.
    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
    """

    num_rewards = len(rewards)
    discounted_rewards = [None]*num_rewards

    for i in range(num_rewards):
        if i == 0:
            # base case, set final timestep's discounted reward
            discounted_rewards[num_rewards - 1] = rewards[num_rewards - 1]
        else:
            # use the discounted reward of the next entry to calculate current
            disc_reward = rewards[num_rewards - 1 - i] + discount_factor * discounted_rewards[num_rewards - i]
            discounted_rewards[num_rewards - 1 - i] = disc_reward
    
    return discounted_rewards


def generate_trajectory(env, model, get_video=False):
    """
    Generates lists of states, actions, and rewards for one complete episode.
    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
    states = []
    actions = []
    rewards = []
    state = env.reset()
    state = compress(state)

    print("shape of state: {}\n".format(tf.shape(state)))
    done = False

    # NOOP until green light
    for i in range(88):
        (obs, rew, end, info) = env.step([0, 0, 0, 0, 0])

    for i in range(300):
        # 1) use model to generate probability distribution over next actions
        # 2) sample from this distribution to pick the next action

        # get cnn output; feed state into a CNN -> state vector

        probabilities = tf.squeeze(model.call(tf.expand_dims(tf.cast(state, tf.float32), axis=0)))
        probabilities = np.reshape(probabilities.numpy(), [model.num_actions])
        possible_actions = np.arange(model.num_actions)
        action = np.random.choice(possible_actions, 1, p=probabilities)[0]

        discrete_actions = DiscreteActions()
        actual_action = discrete_actions.ACTION_MAP[action][1]

        states.append(state)
        actions.append(actual_action)
        state, rwd, done, _ = env.step(actual_action)
        state = compress(state)
        rewards.append(rwd)

        print("shape of the state: {}\n".format(tf.shape(state)))
        #print("reward: {}\n".format(rwd))

    print("Finished 300 iterations")

    print("Calling observe")

    observe(tf.convert_to_tensor(states).numpy())

    print("Finished observing")

    return states, actions, rewards


def train(env, model):
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one
    episode (lists of states, action_probs, and rewards seen/taken in the episode), and
    then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode
    :param env: The openai gym environment
    :param model: The model
    :returns: The total reward for the episode
    """

    # 1) Use generate trajectory to run an episode and get states, actions, and rewards.
    # 2) Compute discounted rewards.
    # 3) Compute the loss from the model and run backpropagation on the model.

    total_reward = 0

    with tf.GradientTape() as tape:
        states, actions, rewards = generate_trajectory(env, model)
        discounted_rewards = discount(rewards)
        loss = model.loss(np.array(states), actions, discounted_rewards)
        total_reward += np.sum(rewards)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_reward

def main():

    env = gym.make("Mario-Kart-Luigi-Raceway-v0")
    state_size = env.observation_space.shape[0]
    discrete_actions = DiscreteActions()
    num_actions = discrete_actions.get_action_space().n
    

    # Initialize model
    model = DK64Model(state_size, num_actions)
    
    # 1) Train your model for 650 episodes, passing in the environment and the agent. 
    # 2) Append the total reward of the episode into a list keeping track of all of the rewards. 
    # 3) After training, print the average of the last 50 rewards you've collected.

    rewards = []
    for i in range(100):
        if (i % 25) == 0:
            print("Train episode {}!\n".format(i))
        reward = train(env, model)
        rewards.append(reward)

    avg_last_rewards = np.sum(rewards[-10:]) / 10
    print("Average of last 10 rewards: {}\n".format(avg_last_rewards))

    # Visualize your rewards.
    # visualize_data(rewards) # commented out as this causes a segfault on my machine


if __name__ == '__main__':
    main()


# ABSTRACT METHODS
# _step(action)
# _observe()
# _render()
# _navigate_menu()
# _get_reward()
# _evaluate_end_state()
# _reset()

# MARIO KART METHODS
# _navigate_post_race_menu()
# _set_character(character)
# _get_lap()

# ENVIRONMENT METHODS
# gym.make('map')
# env.step([arr])
# env.reset()
# env.close()
