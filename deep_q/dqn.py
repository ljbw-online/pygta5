from __future__ import absolute_import, division, print_function

import multiprocessing
import os
import pickle
import shutil
from multiprocessing import Queue, Process
from time import sleep

import numpy as np
import cv2

import reverb
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import TFPyEnvironment
from tf_agents.policies import PolicySaver, PyTFEagerPolicy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
# from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.networks.q_network import QNetwork
from tf_agents.networks.q_rnn_network import QRnnNetwork
from tf_agents.trajectories.policy_step import PolicyStep

from deep_q.evaluate_policy import compute_avg_return
from deep_q.breakout_pyenv import Env, env_name, gamma, max_return

# wn = 'OpenCV'
# cv2.namedWindow(wn)
# Set up a virtual display for rendering OpenAI gym environments.
# display = pyvirtualdisplay.Display(visible=True, size=(700, 500)).start()

# num_iterations = 3_000  # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 4  # @param {type:"integer"}
replay_buffer_max_length = 250_000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 2.5e-4  # @param {type:"number"}
log_interval = 10_000  # @param {type:"integer"}
eval_interval = 10 * log_interval  # @param {type:"integer"}

# num_eval_episodes = 10  # @param {type:"integer"}

# env_name = 'CartPole-v0'
# env_name = 'BreakoutNoFrameskip-v4'
# env = suite_gym.load(env_name)


# def create_q_net():
#     return QRnnNetwork(
#         train_env.observation_spec(),
#         train_env.action_spec(),
#         preprocessing_layers=tf.keras.layers.Rescaling(1. / 255),
#         # preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
#         conv_layer_params=((8, 8, 2), (16, 4, 2), (32, 3, 2)),
#         input_fc_layer_params=(256,),
#         lstm_size=(256,)
#     )


# DON'T FORGET RESCALING LAYER
# q = create_q_net()
# q.create_variables(input_tensor_spec=(tensor_spec.TensorSpec((84, 84, 1), tf.int8, 'observation')))
# q.summary()
# print(q(np.random.randint(0, 255, (1, 210, 160, 3), dtype=np.uint8)))


# def create_cartpole_q_net():
#     return QNetwork(train_env.observation_spec(), train_env.action_spec())
#     return QRnnNetwork(
#         train_env.observation_spec(),
#         train_env.action_spec(),
#         lstm_size=(128,),
#     )


def visualise(input_queue, output_queue):
    num_episodes = 300
    eval_py_env = Env()
    eval_env = TFPyEnvironment(eval_py_env)  # For some reason it needs to be a TFPyEnv when loading policy from disc

    input_queue.get(block=True)
    visualise_policy = tf.saved_model.load(env_name)

    while True:
        try:
            print('computing average')
            average_return = compute_avg_return(eval_env, visualise_policy, num_episodes=num_episodes)

            print(f'Average return over {num_episodes} episodes: {average_return}')

            max_return_achieved = average_return == max_return
            output_queue.put(max_return_achieved)
            if max_return_achieved:
                input_queue.cancel_join_thread()
                output_queue.cancel_join_thread()
                break

            input_queue.get(block=True)

            total_return = 0
            episode_count = 0
            visualise_policy = tf.saved_model.load(env_name)
        except Exception as e:
            print('Exception in subprocess:')
            print(e)


# def compute_avg_return(environment, policy, num_episodes=10, render=False):
#     total_return = 0.0
#
#     # PyTFEagerPolicy makes a TFPolicy compatible with PyEnvironments. Without this, this function runs very slowly and
#     # we get warnings about tf.function retracing.
#     if isinstance(policy, GreedyPolicy):
#         policy = py_tf_eager_policy.PyTFEagerPolicy(policy, use_tf_function=True)
#
#     for _ in range(num_episodes):
#         time_step_local = environment.reset()
#         policy_state = policy.get_initial_state(environment.batch_size)
#         # policy_state = tf.expand_dims(policy_state, axis=0)
#         episode_return = 0.0
#
#         while not time_step_local.is_last():
#             if render:
#                 environment.render(mode='human')
#                 cv2.waitKey(1)
#
#             policy_step = policy.action(time_step_local, policy_state)
#             policy_state = policy_step.state
#             if random_eval_action is not None:
#                 if np.random.random() < 0.05:
#                     action_tensor = tf.constant(random_eval_action, dtype=tf.int32)
#                     policy_step = PolicyStep(action=action_tensor, state=policy_state, info=())
#
#             time_step_local = environment.step(policy_step.action)
#             episode_return += time_step_local.reward
#
#         total_return += episode_return
#
#     avg_return_local = total_return / num_episodes
#     return avg_return_local[0]
    # return avg_return_local


def main():
    train_py_env = Env()
    train_env = TFPyEnvironment(train_py_env)
    eval_py_env = Env()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    observation_sequence_length = 2

    q_net = train_py_env.create_q_net(recurrent=False)

    train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(), train_env.action_spec(), q_network=q_net, optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_huber_loss, train_step_counter=train_step_counter,
        target_update_period=10_000, gamma=gamma, epsilon_greedy=1.0,
    )

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)
    # Reset the train step.
    agent.train_step_counter.assign(0)
    agent.initialize()

    # agent.collect_data_spec just specifies action, observation etc. It doesn't mention sequence length.
    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=observation_sequence_length,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client,
        table_name,
        sequence_length=observation_sequence_length)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=32,  # must be <= cycle_length which is 32 (whatever cycle_length is)
        sample_batch_size=batch_size,
        num_steps=observation_sequence_length).prefetch(batch_size)

    iterator = iter(dataset)

    collect_driver = py_driver.PyDriver(
        train_py_env,
        PyTFEagerPolicy(agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=collect_steps_per_iteration)

    multiprocessing.set_start_method('spawn', force=True)
    visualise_input_queue = Queue()
    visualise_output_queue = Queue()
    # visualise_output_queue.put(False)  # Stops us from waiting before the first average has been computed
    p = Process(target=visualise, args=(visualise_input_queue, visualise_output_queue))
    # p.start()

    checkpoint_dir = env_name + '_checkpoint'
    best_score_filename = env_name + '_best_score'
    best_score_so_far = 0

    if os.path.isdir(checkpoint_dir):
        choice = input('Delete checkpoint and policy? (y/N)')

        if choice == 'y':
            try:
                # rmtree to used with care.
                shutil.rmtree(checkpoint_dir)
                shutil.rmtree(env_name)
            except FileNotFoundError:
                pass
        else:
            with open(best_score_filename, 'rb') as best_score_file:
                best_score_so_far = pickle.load(best_score_file)

    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=train_step_counter
    )

    policy_saver = PolicySaver(agent.policy)
    train_checkpointer.initialize_or_restore()

    time_step = train_py_env.reset()
    policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)

    eval_episodes = 40

    for _ in range(observation_sequence_length - 1):
        collect_driver.run(time_step, policy_state=policy_state)

    while True:
        time_step, _ = collect_driver.run(time_step, policy_state=policy_state)

        experience, unused_info = next(iterator)

        # It looks like for a stack len of e.g. 3 frames, we need experience.observation to have shape [Bx4x...].
        # This gets turned into a batch of Transitions where each step.observation is the first three frames and each
        # next_step.observation is the last three frames.
        train_loss = agent.train(experience).loss

        step = int(agent.train_step_counter)

        if step % log_interval == 0:
            # if visualise_output_queue.empty():
            #     if not p.is_alive():  # can be reported as alive even if it's crashed due to an exception
            #         print('Reinitialising subprocess')
            #         p = Process(target=visualise, args=(visualise_input_queue, visualise_output_queue))
            #         p.start()
            #         visualise_input_queue.put(True)
            # else:
            #     if visualise_output_queue.get(block=True):
            #         print('Max return achieved, exiting.')
            #         p.join()
            #         break
            #     else:
            #         visualise_output_queue.put(False)  # Put the False back on the queue for the next if statement.
            #         sleep(0.1)  # "there may be an infinitesimal delay before the queue’s empty() method returns False"

            print('step = {0}: loss = {1}'.format(step, train_loss))

        # if step == eval_interval:
        #     print('Saving policy and checkpoint')
        #     policy_saver.save(env_name)
        #     train_checkpointer.save(train_step_counter)
        #     visualise_input_queue.put(True)
        # elif step % eval_interval == 0:
        if step % eval_interval == 0:
            average_return = compute_avg_return(eval_py_env, agent.policy, num_episodes=eval_episodes)
            print(f'Average return over {eval_episodes} episodes: {average_return}')

            if average_return > best_score_so_far:
                best_score_so_far = average_return
                print('Saving policy and checkpoint')
                policy_saver.save(env_name)
                train_checkpointer.save(train_step_counter)
                with open(best_score_filename, 'wb') as best_score_file:
                    pickle.dump(best_score_so_far, best_score_file)

            if average_return == max_return:
                print('Max return achieved. Saving policy and exiting.')
                policy_saver.save(env_name)
                break

            # if not visualise_output_queue.empty():
            #     if visualise_output_queue.get(block=True):
            #         print('Max return achieved, exiting.')
            #         p.join()
            #         break
            #     else:
            #         print('Saving policy and checkpoint')
            #         policy_saver.save(env_name)
            #         train_checkpointer.save(train_step_counter)
            #         visualise_input_queue.put(True)
            # else:
            #     pass
            # print('visualise_output_queue empty')


if __name__ == '__main__':
    main()
