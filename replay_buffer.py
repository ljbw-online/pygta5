import os
from collections import deque

import numpy as np

rng = np.random.default_rng()

class ReplayBuffer:
    def __init__(self, env_name, timestep_dtype, batch_size=32, obs_seq_len=4, max_length=1e6):
        self.timestep_dtype = timestep_dtype
        self.batch_size = batch_size
        self.obs_seq_length = obs_seq_len
        self.episodes_stored = 0
        self.episodes = deque()
        self.max_length = max_length
        self.current_length = 0

    def add_episode(self, episode):
        episode = np.array(episode)

        if len(episode) <= self.obs_seq_length:
            print(f'Discarding an episode which is only {len(episode)} timesteps long')
            return

        while (self.current_length + len(episode) > self.max_length):
            self.remove_episode()
            
        self.episodes.append(episode)

        self.current_length += len(episode)
        self.episodes_stored += 1

    def remove_episode(self):
        deleted_episode = self.episodes.popleft()

        self.current_length -= deleted_episode.shape[0]
        self.episodes_stored -= 1

    def get_batch(self):
        observation_sample = []
        observation_next_sample = []
        action_sample = []
        reward_sample = []
        for episode_choice in rng.choice(self.episodes_stored, size=self.batch_size):
            episode = self.episodes[episode_choice]
            # rng.choice(N) returns values >= 0 and < N, which is why we don't get IndexError
            # later when getting obs_next.
            slice_start = rng.choice(len(episode) - self.obs_seq_length)
            slice_end = slice_start + self.obs_seq_length

            obs_slice = slice(slice_start, slice_end)
            obs_next_slice = slice(slice_start + 1, slice_end + 1)

            observation_sample.append(episode[obs_slice]['observation'])

            # slice(0, 4) will get elements 0, 1, 2 and 3. So we want action[slice_end - 1].
            action_sample.append(episode[slice_end - 1]['action'])
            reward_sample.append(episode[slice_end - 1]['reward'])

            observation_next_sample.append(episode[obs_next_slice]['observation'])

        return (np.array(observation_sample), np.array(observation_next_sample), np.array(action_sample),
                np.array(reward_sample))

    def __len__(self):
        # Total number of timesteps stored.
        return self.current_length


def test_replay_buffer():
    import cv2
    from common import put_text, resize
    from environments.secret_sequence import Env
    from trainer import run_episode, QFunction, get_q_net

    key_ord = 0
    action_ords = list(map(ord, map(str, range(10))))
    env = Env()
    action = 0
    terminated = False
    episode = []

    obs_seq_len = 4
    rb = ReplayBuffer(env.name, env.timestep_dtype, obs_seq_len=obs_seq_len, max_length=100)

    observation_shape = env.timestep_dtype['observation'].shape
    input_shape = observation_shape + (obs_seq_len,)

    qfun = QFunction(obs_seq_len, env, get_q_net(input_shape, env.num_actions))

    while True:
        print('adding ep')
        episode, _, _, _ = run_episode(env, qfun)

        rb.add_episode(episode)

        len_sum = sum(map(len, rb.episodes))

        try:
            assert len(rb) == len_sum
        except e:
            print(len(rb), len_sum)
            raise e

        key_ord = ''
        while key_ord != ord('n'):
            obs_sample, obs_next_sample, action_sample, reward_sample = rb.get_batch()

            resize_obs = obs_sample.shape[2] < 84
            for i, obs in enumerate(obs_sample):
                obs_next = obs_next_sample[i]

                if resize_obs:
                    resized_obs = []
                    resized_obs_next = []

                    for frame in obs:
                        resized_obs.append(resize(frame, width=84))

                    for frame in obs_next:
                        resized_obs_next.append(resize(frame, width=84))

                    obs = np.array(resized_obs)
                    obs_next = np.array(resized_obs_next)

                # After the resize but before the hstack
                action = np.zeros_like(obs[0])
                reward = np.zeros_like(obs[0])
                put_text(action, str(action_sample[i]), position=(4, 28), size=0.6)
                put_text(reward, str(reward_sample[i]), position=(4, 28), size=0.6)

                obs = np.hstack(obs)
                obs_next = np.hstack(obs_next)

                obs = np.hstack((obs, action))
                obs_next = np.hstack((reward, obs_next))

                cv2.imshow('obs_seq', np.vstack((obs, obs_next)))

                key_ord = cv2.waitKey(0)

                if key_ord == ord('q'):
                    cv2.destroyAllWindows()
                    exit()
                elif key_ord == ord('n'):
                    break


if __name__ == '__main__':
    test_replay_buffer()
