import os
from collections import deque

import numpy as np

rng = np.random.default_rng()

# We start to get OSError at about 1000 open files.

class ReplayBuffer:
    def __init__(self, directory, env_name, timestep_dtype, batch_size=32, obs_seq_len=4, max_length=1e6, 
                 max_memory_usage=16e9, max_episodes=900):
        # self.directory = os.path.join(directory, env_name + "_replay_buffer")
        self.directory = directory

        try:
            os.makedirs(self.directory)
        except FileExistsError:
            pass
        
        self.timestep_dtype = timestep_dtype
        self.itemsize = timestep_dtype.itemsize
        self.batch_size = batch_size
        self.obs_seq_length = obs_seq_len
        self.episodes_stored = 0
        self.episode_number_high = 0
        self.episode_number_low = 0
        self.episodes = deque()
        self.episode_paths = deque()
        self.episode_memmap_shapes = deque()
        self.reinit_number = 0
        self.max_length = max_length
        self.current_length = 0
        self.max_memory_usage = max_memory_usage
        self.current_memory_usage_upper_bound = 0
        self.max_episodes = max_episodes

        observation_size = timestep_dtype['observation'].itemsize
        action_size = timestep_dtype['action'].itemsize
        reward_size = timestep_dtype['reward'].itemsize
        self.batch_size_in_bytes = observation_size * obs_seq_len + action_size + reward_size

    def add_episode(self, episode):
        episode = np.array(episode)

        if len(episode) <= self.obs_seq_length:
            print(f'Discarding an episode which is only {len(episode)} timesteps long')
            return

        while (self.current_length + len(episode) > self.max_length
               or self.episodes_stored >= self.max_episodes):
            self.remove_episode()
            
        episode_path = os.path.join(self.directory, f'episode_{self.episode_number_high}.npy')
        # print('Adding', episode_path)

        episode_memmap = np.memmap(episode_path, mode='w+', dtype=episode.dtype, shape=episode.shape)
        episode_memmap[:] = episode[:]
        episode_memmap.flush()

        self.episode_paths.append(episode_path)
        self.episodes.append(episode)
        self.episode_memmap_shapes.append(episode.shape)

        self.current_length += len(episode)
        self.episode_number_high += 1
        self.episodes_stored += 1

        # if self.episodes_stored == self.max_episodes:
        #     self.episodes[self.episode_number] = episode_memmap
        #     self.episode_memmap_shapes[self.episode_number] = episode.shape
        # else:
        #     self.episodes.append(episode_memmap)
        #     self.episode_memmap_shapes.append(episode.shape)
        
        # self.episode_number = (self.episode_number + 1) % self.max_episodes
        # self.episodes_stored = min(self.episodes_stored + 1, self.max_episodes)

    def remove_episode(self):
        episode_to_delete_path = self.episode_paths.popleft()
        deleted_shape = self.episode_memmap_shapes.popleft()
        self.episodes.popleft()

        # print('Removing', episode_to_delete_path)
        os.remove(episode_to_delete_path)
        self.current_length -= deleted_shape[0]
        self.episodes_stored -= 1

        self.episode_number_low += 1


    def get_batch(self):
        while self.current_memory_usage_upper_bound + self.batch_size_in_bytes > self.max_memory_usage:
            reinit_path = self.episode_paths[self.reinit_number]
            reinit_memmap_shape = self.episode_memmap_shapes[self.reinit_number]
            self.episodes[self.reinit_number] = np.memmap(reinit_path, mode='r', dtype=self.timestep_dtype,
                                                          shape=reinit_memmap_shape)

            # Keep reinit_number bounded to the number of episodes stored so far
            self.reinit_number = (self.reinit_number + 1) % len(self.episodes)
            self.current_memory_usage_upper_bound -= reinit_memmap_shape[0] * self.itemsize

        observation_sample = []
        observation_next_sample = []
        action_sample = []
        reward_sample = []
        # for episode_choice in rng.choice(self.episodes_stored, size=self.batch_size):
        for random_episode_number in rng.integers(self.episode_number_low, self.episode_number_high, size=self.batch_size):
            episode_index = random_episode_number - self.episode_number_low
            episode = self.episodes[episode_index]
            episode_length = self.episode_memmap_shapes[episode_index][0]
            timestep_start = rng.choice(episode_length - 1 - self.obs_seq_length)
            timestep_slice_end = timestep_start + self.obs_seq_length
            timestep_end = timestep_slice_end - 1
            obs_slice = slice(timestep_start, timestep_slice_end)
            obs_next_slice = slice(timestep_start + 1, timestep_slice_end + 1)

            observation_sample.append(episode[obs_slice]['observation'])
            observation_next_sample.append(episode[obs_next_slice]['observation'])
            action_sample.append(episode[timestep_end]['action'])
            reward_sample.append(episode[timestep_end]['reward'])

        self.current_memory_usage_upper_bound += self.batch_size_in_bytes

        return (np.array(observation_sample), np.array(observation_next_sample), np.array(action_sample),
                np.array(reward_sample))

    def populate_episodes_deque(self):
        for (i, path) in enumerate(self.episode_paths):
            memmap_shape = self.episode_memmap_shapes[i]
            try:
                episode_memmap = np.memmap(path, mode='r', dtype=self.timestep_dtype, shape=memmap_shape)
            except Exception as ex:
              print(ex)
              print(i, f'{path=}', f'{memmap_shape}')
              exit()
            self.episodes.append(episode_memmap)
        
    def __len__(self):
        """Total number of timesteps stored."""
        # return sum(map(lambda shape: shape[0], self.episode_memmap_shapes))
        return self.current_length


def test_replay_buffer():
    import cv2
    from common import put_text, resize
    from environments.secret_sequence import Env

    key_ord = 0
    action_ords = list(map(ord, map(str, range(10))))
    env = Env()
    action = 0
    terminated = False
    episode = []

    obs_seq_len = 4
    rb = ReplayBuffer("~/var/Data", env.name, env.timestep_dtype, obs_seq_len=obs_seq_len, max_length=120)

    while True:
        episode.clear()
        observation = env.reset()

        print(rb.episode_memmap_shapes)
        print(len(rb))

        for _ in range(env.max_steps_per_episode):
            if terminated:
                terminated = False
                break

            cv2.imshow('Observation', observation)
            key_ord = cv2.waitKey(1)

            if key_ord == ord('q'):
                cv2.destroyAllWindows()
                exit()
            # else:
            #     try:
            #         action = action_ords.index(key)
            #     except ValueError:
            #         pass

            action = rng.choice(env.num_actions)
            observation_next, reward, terminated = env.step(action)

            # print(reward)

            episode.append(np.array((observation, action, reward), dtype=env.timestep_dtype))

            observation = observation_next

        rb.add_episode(episode)

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

                # if reward_sample[i] > 0:
                if True:
                    key_ord = cv2.waitKey(0)
                else:
                    key_ord = cv2.waitKey(1)

                if key_ord == ord('q'):
                    cv2.destroyAllWindows()
                    exit()
                elif key_ord == ord('n'):
                    break

# import os
# import pickle

# from dqn import PickleableTrainingState

# path = '/home/ljbw/Data/GTA/training_state'

def remove_episodes_from_deques():
    with open(path, 'r+b') as readfile:
        pstate = pickle.load(readfile)

    rb = pstate.replay_buffer

    for _ in range(2):
        rb.episode_memmap_shapes.popleft()
        rb.episode_paths.popleft()
        rb.episodes_stored -= 1
        rb.episode_number_low += 1

    print(rb.episodes_stored)
    print(len(rb.episode_memmap_shapes))
    print(len(rb.episode_paths))
    print(rb.episode_number_low)
    print(rb.episode_number_high)

    for i in range(3):
        print(rb.episode_paths[i])

    choice = input('Save? ')
    if choice == 'y':
        print('Saving')
        with open(path, 'w+b') as writefile:
            pickle.dump(pstate, writefile)  
    else:
        print('Did not save')


def delete_excess_episodes():
    with open(path, 'r+b') as readfile:
        pstate = pickle.load(readfile)

    rb = pstate.replay_buffer

    # for i in range(3):
    #     print(rb.itemsize)
    for i, shape in enumerate(rb.episode_memmap_shapes):
        if shape != (1000,):
            print(i, shape, rb.episode_paths[i])

    # When viewing a deque item always use indexing rather than popleft, so that you don't remove the item
    print(rb.episodes_stored, rb.episode_paths[0], rb.episode_number_low, rb.episode_number_high)

    # while rb.episodes_stored > 900:
    #     episode_to_delete_path = rb.episode_paths.popleft()
    #     print('deleting', episode_to_delete_path)
    #     deleted_shape = rb.episode_memmap_shapes.popleft()
    #     # rb.episodes.popleft()

    #     os.remove(episode_to_delete_path)
    #     rb.current_length -= deleted_shape[0]
    #     rb.episodes_stored -= 1

    #     rb.episode_number_low += 1

    rb.episode_paths.appendleft('/home/ljbw/Data/GTA/replay_buffer/GTA_replay_buffer/episode_196.npy')

    # When viewing a deque item always use indexing rather than popleft, so that you don't remove the item
    print(rb.episodes_stored, rb.episode_paths[0], rb.episode_number_low, rb.episode_number_high)

    choice = input('Save? ')
    if choice == 'y':
        print('Saving')
        with open(path, 'w+b') as writefile:
            pickle.dump(pstate, writefile)  
    else:
        print('Did not save')


if __name__ == '__main__':
    test_replay_buffer()
