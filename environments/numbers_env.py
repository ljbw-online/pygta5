from collections import deque

import numpy as np
import cv2

# File can't be called 'numbers.py' because Python gets confused about imports

# A previous version of this env had a progress bar at the top of the observation. This
# seemed to significantly slow down training, I assume because it increased the number of
# distinct observations.

# With a delay of 3 the agent has to choose the action displayed 3 frames ago. In other
# words the first frame in a 4-frame deque. If sparsity is 1 then reward will be 1.0 if
# the action is correct and 0.0 otherwise. If sparsity is e.g. 4 then every 4 steps the
# reward will be (num of correct answers in last 4 steps) / 4.

env_name = 'numbers'
gamma = 0.1
epsilon_max = 0.1

random_eval_action = None

action_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

sparse_reward_sequences_per_episode = 5
max_return = sparse_reward_sequences_per_episode * np.float32(1.0)


class Env:
    def __init__(self, eval_mode=False, num_actions=2, sparsity=1, delay=3):
        if num_actions < 1 or num_actions > 10:
            raise ValueError('num_actions must be between 1 and 10 inclusive')

        if sparsity < 1:
            raise ValueError('Minimum sparsity value is 1')

        if delay < 0:
            raise ValueError('Minimum action_delay value is 0')

        self.name = env_name
        # obs shape corresponds to what will go into replay buffer
        self.timestep_dtype = np.dtype(
            [('observation', np.uint8, (84, 84)), ('action', np.int32), ('reward', np.float32)])
        self.input_shape = self.timestep_dtype['observation'].shape
        self.num_actions = num_actions
        self.max_steps_per_episode = 100
        self.evaluation_epsilon = 0.0

        # self.observation_sequence_length = action_delay + 3
        self.eval_mode = eval_mode
        self.current_observation = None
        self.sparsity = sparsity

        self.step_count = 0
        self.delay = delay
        self.previous_numbers = deque(maxlen=delay + 1)
        self.correct_count = np.float32(0)
        self.images = np.zeros((num_actions,) + self.input_shape, dtype=np.uint8)

        for i, image in enumerate(self.images):
            cv2.putText(image, str(i), (2, 76), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (255,), 1)

    def reset(self):
        self.step_count = 0
        self.correct_count = np.float32(0)
        self.previous_numbers.clear()

        self.images[:, 0, 1:] = 0

        random_number = np.random.randint(self.num_actions)

        for _ in range(self.delay + 1):
            self.previous_numbers.append(random_number)

        observation = self.images[random_number]
        self.current_observation = observation

        if self.eval_mode:
            print(f'New episode,   randomly selected number: {random_number}')

        return observation

    def step(self, action):
        self.step_count += 1

        delayed_number = self.previous_numbers.popleft()

        action_was_correct = action == delayed_number

        if self.step_count > self.delay:
            self.correct_count += np.float32(action_was_correct)

        random_number = np.random.randint(self.num_actions)

        self.previous_numbers.append(random_number)
        observation = self.images[random_number]

        self.current_observation = observation

        reward = np.float32(0)
        if self.step_count >= (self.delay + self.sparsity):
            if (self.step_count - self.delay) % self.sparsity == 0:
                reward = self.correct_count / self.sparsity
                self.correct_count = np.float32(0)

        if self.eval_mode:
            print(
                f'received action: {action}, '
                f'number selected {self.delay + 1} steps ago: {delayed_number}, '
                f'hence action_was_correct: {action_was_correct}, and reward: {reward}\n'
                f'number selected on this step: {random_number}, '
            )

        return observation, reward, False

    def render(self, mode=None):
        if mode == 'rgb_array':
            return self.current_observation
        elif mode == 'namedWindow':
            cv2.namedWindow('Numbers')
        else:
            frame = self.current_observation
            cv2.imshow(env_name, frame)
            cv2.waitKey(1)

    def pause(self):
        return
    
    def close(self):
        cv2.destroyAllWindows()


def test_env():
    action_ords = list(map(ord, map(str, range(10))))
    env = Env(eval_mode=True)
    window_name = 'observation'
    cv2.namedWindow(window_name)
    key_ord = 0
    action = 0
    terminated = False
    observation = env.reset()
    while key_ord != ord('q'):
        if terminated:
            observation = env.reset()

        cv2.imshow(window_name, observation)
        key_ord = cv2.waitKey(0)

        try:
            action = action_ords.index(key_ord)
        except ValueError:
            pass

        observation_next, reward, terminated = env.step(action)

        observation = observation_next

    env.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_env()
