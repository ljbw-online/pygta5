import os

import numpy as np

from common import data_dir

rng = np.random.default_rng()


class ReplayBuffer:
    def __init__(self, env_name, timestep_dtype):
        self.directory = os.path.join(data_dir, env_name + '_replay_buffer')
        self.timestep_dtype = timestep_dtype
        self.max_episodes = 10
        self.episodes_stored = 0
        self.episodes = []
        self.episode_number = 0
        self.episode_memmap_shapes = []
        self.batch_number = 0

    def add_episode(self, episode):
        episode = np.array(episode)
        episode_path = os.path.join(self.directory, f'episode_{self.episode_number}')
        episode_memmap_shape = (len(episode),)
        self.episode_memmap_shapes[self.episode_number] = episode_memmap_shape
        np.save(episode_path, episode)
        self.episodes[self.episode_number] = np.memmap(episode_path, mode='r', dtype=self.timestep_dtype,
                                                       shape=episode_memmap_shape)

        self.episode_number = (self.episode_number + 1) % self.max_episodes
        self.episodes_stored = min(self.episodes_stored + 1, self.max_episodes)

    def get_batch(self):
        batch_size = 32
        observation_sample = []
        observation_next_sample = []
        action_sample = []
        reward_sample = []
        for episode_choice in rng.choice(self.episode_number, size=batch_size):
            episode_length = self.episode_memmap_shapes[episode_choice][0]
            timestep_choice = rng.choice(episode_length - 1)
            observation_sample.append(self.episodes[episode_choice][timestep_choice]['observation'])
            observation_next_sample.append(self.episodes[episode_choice][timestep_choice + 1]['observation'])
            action_sample.append(self.episodes[episode_choice][timestep_choice]['action'])
            reward_sample.append(self.episodes[episode_choice][timestep_choice]['reward'])

        # Re-initialise one memmap every batch to keep memory usage low
        episode_path = os.path.join(self.directory, f'episode_{self.batch_number}')
        episode_memmap_shape = self.episode_memmap_shapes[self.batch_number]
        self.episodes[self.batch_number] = np.memmap(episode_path, mode='r', dtype=self.timestep_dtype,
                                                     shape=episode_memmap_shape)

        # batch_number should be bounded to the number of episodes stored so far
        self.batch_number = (self.batch_number + 1) % self.episodes_stored

        return (np.array(observation_sample), np.array(observation_next_sample), np.array(action_sample),
                np.array(reward_sample))
