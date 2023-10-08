import pickle
from math import floor
from multiprocessing import Process, Queue
from time import sleep

from windows import get_keys
from experiments.deep_q_in_stages import training_params_filename, model_path, train_model, create_q_model, epoch_size
#    ^ make sure this is importing from the right env and model_format


def train():
    with open(training_params_filename, 'rb') as training_state_file:
        training_state = pickle.load(training_state_file)

    max_memory_length = training_state['max_memory_length']
    data_count = min(training_state['data_count'], max_memory_length)

    count_q = Queue()
    weights_q = Queue()
    model = create_q_model()
    model.summary()
    training_stage_count = 0

    training_timeout = 1000

    get_keys()
    num_training_stages = floor(data_count / epoch_size)
    print(f'training for {num_training_stages} stages')
    for _ in range(num_training_stages):
        print(f'training_stage_count: {training_stage_count}')
        five = False

        training_process = Process(target=train_model, args=(count_q, weights_q))
        training_process.start()

        count_q.put(training_stage_count)
        count_q.put(model.get_weights())

        for i in range(training_timeout):
            five = five or '5' in get_keys()
            if i == training_timeout - 1:
                choice = input('Didn\'t get weights from subprocess')
                five = choice == '5'
            elif not training_process.is_alive():
                choice = input('training_process died')
                five = choice == '5'
                break
            elif weights_q.empty():
                sleep(1)
            else:
                model.set_weights(weights_q.get(block=True))
                break

        training_process.join(timeout=10)

        training_stage_count += 1

        if five:
            print('five')

        if training_stage_count == num_training_stages:
            print(f'tsc == nts: {training_stage_count}, {num_training_stages}')

        if five or training_stage_count == num_training_stages:
            choice = input('save_vis / quit ?')
            if choice == 's':
                model.save(model_path)
                break
            elif choice == 'q':
                break


if __name__ == '__main__':
    train()
