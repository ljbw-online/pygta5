import cv2
import numpy as np
import tensorflow as tf
from tf_agents.environments import TFPyEnvironment
from tf_agents.policies import PyTFEagerPolicy, GreedyPolicy
from tf_agents.trajectories.policy_step import PolicyStep

from pyenvs.breakout import env_name, Env


def visualise():
    eval_env = Env()
    policy = tf.saved_model.load(env_name)

    while True:
        avg_return = compute_avg_return(eval_env, policy)

        if avg_return is not None:
            print(f'Average return: {avg_return}')
        else:
            print('Returning')
            return


def compute_avg_return(py_environment, tf_policy, num_episodes=10, single_step=False, render=False):
    total_return = 0.0

    if single_step:
        wait_key_duration = 0
    else:
        wait_key_duration = 1

    # When we lose a life in Breakout we need to press action 1 to start the next life. Often the agent doesn't choose
    # action 1 in response to the lost-life screen and hence gets indefinitely stuck. To work around this, if the env
    # has a random_eval_action we inject it with a low probability.
    try:
        random_eval_action = py_environment.random_eval_action
    except AttributeError:
        random_eval_action = None

    environment = TFPyEnvironment(py_environment)

    # If the policy has been initialised in the calling script then we need to wrap it in PyTFEagerPolicy otherwise this
    # function runs very slowly and we get warnings about function retracing. If the policy has been loaded from disc
    # then we can't do this because of a weird exception.
    if isinstance(tf_policy, GreedyPolicy):
        policy = PyTFEagerPolicy(tf_policy, use_tf_function=True)
    else:
        policy = tf_policy

    for _ in range(num_episodes):
        time_step_local = environment.reset()
        policy_state = policy.get_initial_state(batch_size=1)
        episode_return = 0.0

        while not time_step_local.is_last():
            if single_step or render:
                environment.render(mode='human')

            if cv2.waitKey(wait_key_duration) == ord('q'):
                cv2.destroyAllWindows()
                return None

            policy_step = policy.action(time_step_local, policy_state)
            policy_state = policy_step.state

            if random_eval_action is not None:
                if np.random.random() < 0.05:
                    policy_step = PolicyStep(action=tf.constant(random_eval_action, dtype=tf.int32), state=policy_state,
                                             info=())

            time_step_local = environment.step(policy_step.action)
            episode_return += time_step_local.reward

        if single_step:
            print(f'Episode return: {episode_return.numpy()[0]}')

        total_return += episode_return

    avg_return_local = total_return / num_episodes
    return avg_return_local.numpy()[0]


if __name__ == '__main__':
    # single_step = True
    single_step = False
    env = Env()

    if single_step:
        env.render(mode='namedWindow')

    average_return = compute_avg_return(env, tf.saved_model.load(env_name), num_episodes=40, single_step=single_step,
                                        render=True)

    if average_return is not None:
        print(average_return)

        if single_step:
            cv2.destroyWindow(env_name)
