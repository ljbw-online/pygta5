from numpy import mean

from tf_agents.policies.random_tf_policy import RandomTFPolicy

from evaluate_policy import compute_average_return
from environments.secret_sequence import Env


if __name__ == '__main__':
    episode_returns = []
    env = Env()
    policy = RandomTFPolicy(env.time_step_spec(), env.action_spec())

    episode_count = 0
    episode_increment = 256

    while True:
        for _ in range(episode_increment):
            episode_returns.append(compute_average_return(env, policy, num_episodes=1, render_env=False))
            episode_count += 1

        average_return = mean(episode_returns)

        print(f'Episode: {episode_count}, average return: {average_return}')
