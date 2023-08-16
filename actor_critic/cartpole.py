import os
from time import sleep, time

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from common import get_keys

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
env = gym.make("CartPole-v1", render_mode='human')  # Create the environment
env.action_space.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

num_inputs = 4
num_actions = 2
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

if os.path.isdir(''):
    choice = input('Load model? (Y/n)')
    if choice != 'n':
        model = keras.models.load_model('')
        model.compile()

        while True:
            start_time = time()
            state, info = env.reset()
            env.render()

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            action_probs, critic_value = model(state)
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))

            state, reward, terminated, truncated, info = env.step(action)

            sleep(0.05)
            if (terminated or truncated) or ('5' in get_keys()):
                exit()

            # while time() < start_time + 0.013:
            #     sleep(0.001)

while True:  # Run until solved
    state, info = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            env.render()  # Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))
            # LJBW: the log of the probability is taken in order to extremely disincentivise low probabilities for
            # actions which went on to result in a higher reward than the Critic predicted (see below).
            # They probably use TensorFlow's log instead of Python's log so that log_prob below is a tensor, which in
            # turn allows loss_value to be a tensor, allowing tape.gradient to be used.

            # Apply the sampled action in our environment
            state, reward, terminated, truncated, info = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward
            # LJBW: state is [cart position, cart velocity, pole angle, pole angular velocity]

            if terminated or truncated:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward  # LJBW: why not just sum of rewards?

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma  # LJBW: rewards in the *future* no?
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:  # [::-1] reverses a list
            discounted_sum = r + gamma * discounted_sum  # LJBW: why do we "discount" the reward sum?
            returns.insert(0, discounted_sum)
            # LJBW: At each timestep what was the total reward THAT WAS GOING TO BE received after that timestep
            # (including the reward from the timestep in question)
            # If gamma == 1 then returns is the reverse of the cummulative sum of the reverse of rewards_history.
            # If rewards_history == [1,2,3] then discouted_sum ends up being:
            # [1 + g*(2+g*(3+g*0), 2 + g*(3+g*0), 3 + g*0] where g == gamma
            # The Critic network is trying to predict at each step the total reward that will be received
            # from now onwards. But the reward values are "discounted" according to how far in the future they are.
            # So the Critic benefits most from predicting the near future correctly.

        # LJBW Experimentation
        discounted_returns = returns.copy()

        non_discounted_returns = []
        non_discounted_sum = 0
        for r in rewards_history[::-1]:
            non_discounted_sum = r + 1.0 * non_discounted_sum
            non_discounted_returns.insert(0, non_discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)  # LJBW: why do we normalise the deviations?
        returns = returns.tolist()
        # LJBW: This may be similar to batch normalisation. Adding the float32 epsilon is probably to account for
        # episodes in which the rewards are very invariable.

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss
            # If we received more reward than was predicted then diff will be positive. If the action which resulted
            # in that difference was given a low probability then -log_prob will also be positive and large.
            # If the probability of the selected action was high then whatever diff is we don't get a big contribution
            # to loss.
            # If the probability is low but we received less reward than predicted then we get a negative contribution
            # to loss
            # The Actor is strongly incentivised to not give low probabilities to actions which result in more reward
            # than the Critic is predicting.

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))

        error_sum = 0
        for non_disc_sum, disc_sum, norm_ret, prediction in zip(non_discounted_returns, discounted_returns, returns, critic_value_history):
            # print('sum: {:.3}, disc_sum: {:.3}, norm_disc: {:.3}, pred: {:.3}'.format(non_disc_sum, disc_sum, norm_ret, prediction))
            error_sum += abs(norm_ret - prediction)

        print('error_sum/{}: {:.3}'.format(len(critic_value_history), error_sum.numpy()/len(critic_value_history)))

        # choice = input()
        # if choice == '5':
        #     exit()

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)  # loss_value is a Tensor!
        print(sum(critic_losses).numpy())
        grads = tape.gradient(loss_value, model.trainable_variables)  # model.trainable_variables is a point in sol spac
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 30:  # 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        model.save('actor-critic')
        break
