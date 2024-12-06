import keras

from dqn import Env, average_return, best_model_path
from q_networks import QFunction


if __name__ == '__main__':
    env = Env()
    average_return = average_return(30, env, QFunction(4, env, keras.models.load_model(best_model_path)))

    print(average_return)
