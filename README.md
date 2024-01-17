The current goal of this project is to understand and reproduce the reinforcement learning work which resulted in 
[Agent57](https://deepmind.google/discover/blog/agent57-outperforming-the-human-atari-benchmark/).

dqn.py is the main file and features a double deep Q-learning algorithm with a dueling-architecture model. The best 
agent so far broke an average of 47.3 bricks over 30 episodes of Breakout (i.e. I have not seen it acheive the max 
score yet).

The original goal was to make a self-driving car in GTA V. I was inspired by Sentdex's original pygta5 project. I made 
a FiveM mod which allows me to send actions to GTA  via a socket. I run the game on a separate Windows PC and get the 
image from the game window with a capture card.