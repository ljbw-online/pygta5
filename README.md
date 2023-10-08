# pygta5
This project has the long term goal of applying reinforcement learning to the task of GTA V driving. The agent will be 
rewarded for staying on the road and avoiding collisions as much as possible.

## Current Approach
The main file is dqn.py. In this file I am trying to use the DQN agent from TF-Agents to solve Breakout with Deep 
Q-Learning. Once I have verified that my script can solve Breakout I will move back to GTA and use that as the 
environment. I have made a FiveM mod which allows me to send actions to GTA via a socket. I run the game on a 
separate Windows PC and get the image from the game window with a capture card.

Despite much experimentation and inspection of the TF-Agents code, I have so far never seen the DQN agent exceed the 
performance of a random policy on Breakout. If you have successfully used TF-Agents to solve a non-trivial environment 
please contact me and let me know how you did it.

Thanks to Sentdex for his initial work on this.