import os
import pickle
import shutil

import reverb
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import TFPyEnvironment
from tf_agents.policies import PolicySaver, PyTFEagerPolicy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from evaluate_policy import compute_avg_return
from pyenvs.breakout import Env, env_name, gamma, max_return

collect_steps_per_iteration = 4
replay_buffer_max_length = 1_000_000

batch_size = 64
learning_rate = 2.5e-4
log_interval = 10_000
eval_interval = 10 * log_interval
num_eval_episodes = 40

train_py_env = Env()
train_env = TFPyEnvironment(train_py_env)
eval_py_env = Env()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

# If the qnet consumes a sequence of N frames then this needs to be N + 1 for Deep Q-learning.
observation_sequence_length = 2

q_net = train_py_env.create_q_net(recurrent=False)

train_step_counter = tf.Variable(0)

# As far as I can tell from the code, this agent does not gradually reduce the value of epsilon. This means that if
# epsilon_greedy starts off at 1.0 then the exploration will remain completely random.
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_huber_loss,
    train_step_counter=train_step_counter,
    target_update_period=10_000,
    gamma=gamma,
    epsilon_greedy=1.0,
)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)
agent.initialize()

# agent.collect_data_spec just specifies action, observation etc. It doesn't mention sequence length.
table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

# Reverb initialisation ---
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

dataset = replay_buffer.as_dataset(
    num_parallel_calls=32,  # must be <= cycle_length which is 32
    sample_batch_size=batch_size,
    num_steps=observation_sequence_length).prefetch(batch_size)

dataset_iterator = iter(dataset)
# ---

# An observer is a callable that takes an observation and adds it to the replay buffer.
rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=observation_sequence_length)

collect_driver = py_driver.PyDriver(
    train_py_env,
    PyTFEagerPolicy(agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

checkpoint_dir = env_name + '_checkpoint'
best_score_filename = env_name + '_best_score'

train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step_counter)

policy_saver = PolicySaver(agent.policy)


def save_training_state(best_score_so_far):
    print('Saving policy and checkpoint')
    policy_saver.save(env_name)
    train_checkpointer.save(train_step_counter)
    with open(best_score_filename, 'wb') as best_score_file:
        pickle.dump(best_score_so_far, best_score_file)


def main():
    best_score_so_far = 0
    if os.path.isdir(checkpoint_dir):
        choice = input(f'Delete {env_name} checkpoint and policy? (y/N)')

        if choice == 'y':
            try:
                # rmtree is equivalent to rm -rf. The directory names being passed to it better be correct!
                shutil.rmtree(checkpoint_dir)
                shutil.rmtree(env_name)
            except FileNotFoundError:
                pass
        else:
            with open(best_score_filename, 'rb') as best_score_file:
                best_score_so_far = pickle.load(best_score_file)
    else:
        print('No checkpoint')

    train_checkpointer.initialize_or_restore()

    time_step = train_py_env.reset()
    policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)

    # Put one batch of observations into the replay buffer.
    for _ in range(batch_size + 1):
        collect_driver.run(time_step, policy_state=policy_state)

    while True:
        time_step, _ = collect_driver.run(time_step, policy_state=policy_state)

        experience, unused_info = next(dataset_iterator)

        # It looks like for a stack len of e.g. 3 frames, we need experience.observation to have shape [Bx4x...].
        # This gets turned into a batch of Transitions where each step.observation is the first three frames and each
        # next_step.observation is the last three frames.
        train_loss = agent.train(experience).loss

        step = int(agent.train_step_counter)

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % (eval_interval * 10) == 0:
            save_training_state(best_score_so_far)

        if step % eval_interval == 0:
            average_return = compute_avg_return(eval_py_env, agent.policy, num_episodes=num_eval_episodes)
            print(f'Average return over {num_eval_episodes} episodes: {average_return}')

            if average_return == max_return:
                print('Max return achieved. Saving policy and exiting.')
                policy_saver.save(env_name)
                break
            elif average_return > best_score_so_far:
                best_score_so_far = average_return
                save_training_state(best_score_so_far)


if __name__ == '__main__':
    main()
