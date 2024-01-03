This project has the long term goal of applying reinforcement learning to the task of GTA V driving. The current reward 
function could be expressed as "distance gained since previous time-step". In the future the agent will be incentivised 
to stay on the road and avoid collisions. 

The main file is dqn.py which features a double deep Q-learning algorithm with a dueling-architecture model. A 
single-stream (i.e. non-dueling) model exceeds random performance in Atari Breakout after less than 24 hours of 
training with this script. This is not true for the dueling-architecture model so there must be problems with my 
implementation so far.

I have made a FiveM mod which allows me to send actions to GTA via a socket. I run the game on a 
separate Windows PC and get the image from the game window with a capture card.

Thanks to Sentdex for his initial work on this.