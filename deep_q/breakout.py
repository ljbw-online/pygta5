import multiprocessing
import os

# from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import gymnasium as gym
import numpy as np

from common import get_keys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


num_actions = 4


def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(4, 84, 84,))
    inputs = layers.Rescaling(1./255)(inputs)

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu", data_format='channels_first')(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu", data_format='channels_first')(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu", data_format='channels_first')(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)


# In the Deepmind paper they use RMSProp however the Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
# Using huber loss for stability
loss_function = keras.losses.Huber()
gamma = 0.99  # Discount factor for past rewards
nones = (None, None, None, None, None, None, None)


def update_model(inq, outq):
    model = create_q_model()
    model_target = create_q_model()
    while True:
        # if inq.empty():
        #     sleep(0.1)
        # else:
        (model_weights, model_target_weights, state_sample, state_next_sample, rewards_sample,
            action_sample, done_sample) = inq.get(block=True)

        if model_weights is None:
            break

        # st = time()
        model.set_weights(model_weights)
        model_target.set_weights(model_target_weights)

        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        future_rewards = model_target.predict(state_next_sample, verbose=0)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

        # If final frame set the last value to -1
        updated_q_values = updated_q_values * (1 - done_sample) - done_sample
        # LJBW: updated_q_values contains a -1 whenever it happens that the env was "terminated" on the step that
        # we chose to put into the random sample of steps. Otherwise it contains the Q value.

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, num_actions)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = model(state_sample)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = loss_function(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # print('update_model took {: 4.3}s'.format(time() - st))
        outq.put(model.get_weights())


# update_process = Thread(target=update_model)


def main():
    # Configuration paramaters for the whole setup
    seed = 42
    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    epsilon_interval = (
            epsilon_max - epsilon_min
    )  # Rate at which to reduce chance of random action being taken
    batch_size = 32  # Size of batch taken from replay buffer
    max_steps_per_episode = 10000

    # Use the Baseline Atari environment because of Deepmind helper functions
    # env = make_atari("BreakoutNoFrameskip-v4")
    # Warp the frames, grey scale, stake four frame and scale to smaller ratio
    # env = wrap_deepmind(env, frame_stack=True, scale=True)

    env = gym.make("BreakoutNoFrameskip-v4", render_mode='human')
    env = gym.wrappers.AtariPreprocessing(env)
    env = gym.wrappers.FrameStack(env, 4)
    env.seed(seed)
    # env.metadata['render_fps'] = 60  # doesn't do anything

    # The first model makes the predictions for Q-values which are used to
    # make an action.
    model = create_q_model()
    # Build a target model for the prediction of future rewards.
    # The weights of a target model get updated every 10000 steps thus when the
    # loss between the Q-values is calculated the target Q-value is stable.
    model_target = create_q_model()
    model.summary()

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0
    # Number of frames to take random action and observe output
    epsilon_random_frames = 50000
    # epsilon_random_frames = 0
    # Number of frames for exploration
    epsilon_greedy_frames = 1000000.0
    # epsilon_greedy_frames = 10.0
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 100000
    # Train the model after 4 actions
    update_after_actions = 4
    # How often to update the target network
    update_target_network = 10000

    inq = multiprocessing.Queue()
    outq = multiprocessing.Queue()
    process = multiprocessing.Process(target=update_model, args=(inq, outq))
    outq.put(model.get_weights())  # put something on the queue for the first loop iteration
    process.start()
    frames_skipped = 0

    get_keys()

    while True:  # Run until solved
        state, info = env.reset()
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            # LJBW: env.render() is no longer necessary with render_mode='human' above
            # env.render()  # Adding this line would show the attempts of the agent in a pop up window.
            frame_count += 1

            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                # print('calling model')
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action to our environment
            # LJBW: terminated is something in the MDP like winning or dieing; truncated something outside of the MDP
            # like a time limit being reach or the agent going out of the play area.
            state_next, reward, terminated, truncated, info = env.step(action)
            # state_next = np.array(state_next)  # LJBW: state_next is already a numpy array

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(terminated)
            rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            # if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            if not outq.empty() and len(done_history) > batch_size:
                weights = outq.get()
                # print('skipped {} frames'.format(frames_skipped))
                frames_skipped = 0
                model.set_weights(weights)

                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                update_args = (model.get_weights(), model_target.get_weights(), state_sample, state_next_sample,
                               rewards_sample, action_sample, done_sample)

                inq.put(update_args)

            frames_skipped += 1

            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if terminated:
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        episode_count += 1

        if running_reward > 40:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            print('Saving model and quitting')
            model.save('breakout')
            inq.put(nones)
            break

        if 'X' in get_keys():
            model.save('breakout')
            inq.put(nones)
            break

    process.join()


if __name__ == '__main__':
    main()
