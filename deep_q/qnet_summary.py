from breakout_pyenv import Env

qnet = Env().create_q_net(recurrent=False)
qnet.create_variables()
qnet.summary()
