import tensorflow as tf
from tf_agents.environments import TFPyEnvironment

from dqn import Env, env_name, max_return
from evaluate_policy import compute_average_return

# Code for visualising the agent's gameplay in a subprocess

# This was required to get tf_agents to be imported in a subprocess:
# multiprocessing.set_start_method('spawn', force=True)
# However the subprocess would occasionally crash. Sometimes it was a traceback from tensorflow that looked pretty
# low-level. Other times it was a traceback from multiprocessing which did not cause the main process to crash but which
# I couldn't capture with a try-except in the subprocess.


def visualise(input_queue, output_queue):
    num_episodes = 300
    eval_py_env = Env()

    input_queue.get(block=True)
    visualise_policy = tf.saved_model.load(env_name)

    while True:
        try:
            print('computing average')
            average_return = compute_average_return(eval_py_env, visualise_policy, num_episodes=num_episodes)

            print(f'Average return over {num_episodes} episodes: {average_return}')

            max_return_achieved = average_return == max_return
            output_queue.put(max_return_achieved)
            if max_return_achieved:
                input_queue.cancel_join_thread()
                output_queue.cancel_join_thread()
                break

            input_queue.get(block=True)

            visualise_policy = tf.saved_model.load(env_name)
        except Exception as e:
            print('Exception in subprocess:')
            print(e)
