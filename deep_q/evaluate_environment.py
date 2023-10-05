from numpy import mean

from tf_agents.policies.random_tf_policy import RandomTFPolicy

from deep_q.evaluate_policy import compute_avg_return
from deep_q.secret_sequence import Env

if __name__ == '__main__':
    episode_returns = []
    env = Env()
    policy = RandomTFPolicy(env.time_step_spec(), env.action_spec())

    episode_count = 0
    episode_increment = 256

    while True:
        for _ in range(episode_increment):
            episode_returns.append(compute_avg_return(env, policy, num_episodes=1, render=False))
            episode_count += 1

        average_return = mean(episode_returns)

        print(f'Episode: {episode_count}, average return: {average_return}')
