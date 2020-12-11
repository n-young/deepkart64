import os
import time
import sys
import dill

sys.path.insert(0, "../gym_mupen64plus/envs/MarioKart64")
import gym, gym_mupen64plus
import numpy as np
import tensorflow as tf
from discrete_envs import DiscreteActions
from model import DK64Model
from utils import compress, observe

# Global tweakable parameters.
NUM_EPISODES = 10
EPISODE_LENGTH = 10
BATCH_SZ = 1
SAVE_FREQUENCY = BATCH_SZ * 20
COMPRESS_FACTOR = 5


def discount(rewards, discount_factor=0.99):
    """
    Takes in a list of rewards for each timestep in an episode, and
    returns a list of the discounted rewards for each timestep.
    Refer to the slides to see how this is done.
    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
    """
    # Calculate discounted rewards.
    discounted_rewards = [rewards[-1]]
    for i in range(2, len(rewards) + 1):
        discounted_rewards.insert(
            0, rewards[-i] + discount_factor * discounted_rewards[0]
        )
    return discounted_rewards


def generate_trajectory(env, model, get_video=False):
    """
    Generates lists of states, actions, and rewards for one complete episode.
    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
    # Initialize variables and states.
    states = []
    frames = []
    actions = []
    rewards = []
    current_episode_reward = 0
    state = env.reset()
    done = False

    # NOOP until green light
    for _ in range(88):
        env.step([0, 0, 0, 0, 0])

    # Set up the break condition - runs for maximum of EPISODE_LENGTH seconds.
    start_time = time.time()
    while time.time() < start_time + EPISODE_LENGTH:
        # Break out of loop if episode ended.
        if done:
            break

        # Get probabilities.
        state, original = compress(state, COMPRESS_FACTOR)
        probabilities = model.call(tf.expand_dims(tf.cast(state, tf.float32), axis=0))
        probabilities = tf.squeeze(probabilities)
        probabilities = np.reshape(probabilities.numpy(), [model.num_actions])

        # Select action.
        possible_actions = np.arange(model.num_actions)
        action = np.random.choice(possible_actions, 1, p=probabilities)[0]
        discrete_actions = DiscreteActions()
        actual_action = discrete_actions.ACTION_MAP[action][1]

        # Save state and apply action, take a step.
        states.append(state)
        if get_video:
            frames.append(original)
        actions.append(action)
        state, rwd, done, _ = env.step(actual_action)
        rewards.append(rwd)
        current_episode_reward += rwd

    # Output video if specified.
    if get_video:
        if "-lo" in sys.argv:
            observe(np.array(frames), sys.argv[3])
        if "-o" in sys.argv:
            observe(np.array(frames), sys.argv[2])

    # Return.
    return states, actions, rewards


def train(env, model, get_video=False):
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
    # Generate video.
    if get_video:
        generate_trajectory(env, model, get_video)
        return None

    # Initialize total reward, run simulation, calculate losses.
    losses = []
    total_reward = 0
    with tf.GradientTape() as tape:
        for _ in range(BATCH_SZ):
            states, actions, rewards = generate_trajectory(env, model)
            discounted_rewards = discount(rewards)
            loss = model.loss(
                tf.convert_to_tensor(states).numpy(), actions, discounted_rewards
            )
            losses.append(loss)
            total_reward += np.sum(rewards)
        total_loss = tf.reduce_mean(losses)

    # Apply gradients, return total_reward..
    gradients = tape.gradient(total_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_reward / BATCH_SZ


def main():
    """
    Main function.
    """
    # Initialize environment and important values.
    env = gym.make("Mario-Kart-Luigi-Raceway-v0")
    state_size = env.observation_space.shape[0]
    discrete_actions = DiscreteActions()
    num_actions = discrete_actions.get_action_space().n
    model = DK64Model(state_size, num_actions)

    # Load weights if -l or -ls or -lo flag specified.
    print(sys.argv)
    if "-l" in sys.argv or "-ls" in sys.argv or "-lo" in sys.argv:
        if len(sys.argv) != 3 and "-l" in sys.argv:
            print("CORRECT USAGE: -l <load_from>")
        elif len(sys.argv) != 4 and "-ls" in sys.argv:
            print("CORRECT USAGE: -ls <load_from> <save_to")
        elif len(sys.argv) != 4 and "-lo" in sys.argv:
            print("CORRECT USAGE: -lo <load_from> <video_path>")
        else:
            print("Loading model...")
            file = open(sys.argv[2], "rb")
            model = dill.load(file)
            file.close()
            print("Confirming model loaded correctly...")
            print(model.trainable_variables)
            print("Model variables printed!")
            # If -lo, load and then output and return - no more training.
            if "-lo" in sys.argv:
                train(env, model, get_video=True)
                return None

    # Creates folder to save models to if -S flag specified.
    if "-S" in sys.argv:
        if len(sys.argv) != 3:
            print(
                "CORRECT USAGE: -S <archive_folder> e.g. -S ./save_to (don't have a trailing /)."
            )
        else:
            try:
                os.mkdir(sys.argv[2])
            except OSError as error:
                print("Folder already exists - continuing execution.")

    # Create rewards, train for NUM_EPISODES episodes.
    rewards = []
    for i in range(NUM_EPISODES):
        # Train once.
        reward = train(env, model)
        rewards.append(reward)
        print("Train episode {}! Reward: {}\n".format(i, reward))

        # On every SAVE_FREQUENCY run, save the model if -S flag specified.
        if i % SAVE_FREQUENCY == 0 and "-S" in sys.argv:
            if len(sys.argv) != 3:
                print("CORRECT USAGE: -S <archive_folder>")
            else:
                filename = sys.argv[2] + "/model-" + str(i // SAVE_FREQUENCY) + ".pkl"
                file = open(filename, "wb")
                dill.dump(model, file)
                file.close()

            avg_last_rewards = np.sum(rewards[-SAVE_FREQUENCY:]) / SAVE_FREQUENCY
            print(
                "Average of last {} rewards: {}\n".format(
                    SAVE_FREQUENCY, avg_last_rewards
                )
            )

    # Print average of last 10 rewards.
    if i % 10 == 0 and i >= 10:
        avg_last_rewards = np.sum(rewards[-10:]) / 10
        print("Average of last 10 rewards: {}\n".format(avg_last_rewards))

    # Save model if -s or -ls flag specified.
    if "-s" in sys.argv:
        if len(sys.argv) != 3:
            print("CORRECT USAGE: -s <save_to>")
        else:
            file = open(sys.argv[2], "wb")
            dill.dump(model, file)
            file.close()
    elif "-ls" in sys.argv:
        if len(sys.argv) != 4:
            print("CORRECT USAGE: -ls <load_from> <save_to")
        else:
            file = open(sys.argv[3], "wb")
            dill.dump(model, file)
            file.close()

    # Observe video if -o flag specified. Must load a model too.
    if "-o" in sys.argv:
        if len(sys.argv) != 3:
            print("CORRECT USAGE: -o <video_path>")
        else:
            train(env, model, get_video=True)


# Main.
if __name__ == "__main__":
    main()
