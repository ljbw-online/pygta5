import json
from time import time

# import numpy as np
import keras
from websockets import ConnectionClosedError
# from websockets.sync.server import serve
from websockets.sync.client import connect

from trainer import Env, obs_seq_len, get_q_net, websocket_port, run_episode, QFunction


# def echo(websocket):
#     websocket.send('hello from server')
#
#     env = Env()
#     observation_shape = env.timestep_dtype['observation'].shape
#     input_shape = observation_shape + (obs_seq_len,)
#     model = get_q_net(input_shape, env.num_actions)
#
#     model_config = serialize_keras_object(model)
#     model_config_json = json.dumps(model_config)
#
#     weights = model.get_weights()
#     weights_config = serialize_keras_object(weights)
#     weights_json = json.dumps(weights_config)
#
#     websocket.send(model_config_json)
#     websocket.send(weights_json)
#
#     for message in websocket:
#         if message == 'q':
#             print('quitting server')
#             break
#         else:
#             print('received', message)
#
#
# def run_server():
#     with serve(echo, "localhost", 8765) as server:
#         print('starting server')
#         server.serve_forever()


def client():
    env = Env()
    observation_shape = env.timestep_dtype['observation'].shape
    input_shape = observation_shape + (obs_seq_len,)
    model = get_q_net(input_shape, env.num_actions)
    q_function = QFunction(obs_seq_len, env, model)

    with connect("ws://localhost:" + str(websocket_port), max_size=None) as websocket:
        print('waiting for first msg')
        trainer_msg = websocket.recv()
        print('received')

        trainer_dict = json.loads(trainer_msg)
        epsilon = trainer_dict['epsilon']
        weights = keras.saving.deserialize_keras_object(trainer_dict['weights_config'])

        model.set_weights(weights)

        while True:
            before_ep = time()
            episode, terminated, q_predictions, q_prediction_step_counts = run_episode(env, q_function, epsilon)
            print(f'ran ep in {time() - before_ep}s')

            timestep_tuple_list = episode.tolist()

            for i, (observation, action, reward) in enumerate(timestep_tuple_list):
                timestep_tuple_list[i] = (observation.tolist(), action, reward)

            # Number of q_predictions is less than number of timesteps.
            for i, q_prediction in enumerate(q_predictions):
                q_predictions[i] = q_prediction.tolist()

            tuple_for_server = (timestep_tuple_list, terminated, q_predictions, q_prediction_step_counts)

            json_for_server = json.dumps(tuple_for_server)

            print('sending ep')
            websocket.send(json_for_server)

            # try:
            print('waiting for new weights')
            try:
                trainer_dict = json.loads(websocket.recv())
            except ConnectionClosedError:
                return
            epsilon = trainer_dict['epsilon']
            weights = keras.saving.deserialize_keras_object(trainer_dict['weights_config'])
            model.set_weights(weights)
            # except TimeoutError:
            #     print('no new weights, continuing')


if __name__ == "__main__":
    client()

