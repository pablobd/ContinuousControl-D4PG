# Report

## Solution

The agent is build following the D4PG [paper](https://openreview.net/pdf?id=SyZipzbCb). This is an adapted version of the DDPG algorithm (usually seen as a continuous control action spaces version of Q-learning) that works as follows.

* it has two kinds of networks the Actor (that predicts the action in multidimensional continuous spaces) and the Critic or Q (that estimates the cumulative reward of the episode usually using Temporal Difference methods),
* there are two networks of each kind a local network (updated at each step using gradient descent) and a target network updated softly with the weights of the local networks. This is done to stabilize learning as it can become very unstable,
* during learning noise is added to the Actor output in order to be in a exploratory mode, which is removed on test mode,
* there are several agent that work in independent environment and each one adds its experience to a replay buffer that is shared by all agents and, 
* the (local) actor and critic networks are updated using different samples from the replay buffer.

On this implementation we chose the following strategy to solve the environment,
* the (local) actor and critic networks are updated 20 times in a row (one for each agent), using 20 different samples from the replay buffer (because the environment provided has 20 agents),
* the (target) actor and critic networks are updated once every two updates and with a soft update,
* gradient clipping is used on the critic network to further stabilize learning, 
* the Critic network with 33 input units (state size), hidden layers of 128, 64, 64 + 4 (action size) and 32 units, and an output layer of 1 unit,
* the Actor network has an input layer of 33 units (state size), two hidden layer of 64 and 32 units, and an output layer of 4 units (action size),
* noise dampening, with a function of the average score, so that when the average score approaches the maximum score of 40, the noise amplitude reduces.

The following values of the hyperparameters are used:
* BUFFER_SIZE = 1e6
* BATCH_SIZE = 128
* GAMMA = 0.99
* TAU = 1e-3
* LR_ACTOR = 1e-4
* LR_CRITIC = 3e-4
* WEIGHT_DECAY_CR = 1e-4
* WEIGHT_DECAY_AC = 0
* UPDATE_EVERY = 2

## Results

We have done the training in three stages:

1. Training during 50 episodes and store weights. The average score goes from 0 to 9.
2. Training during 30 episodes and store weights. The average score goes from 4 to 35.
3. Training during 100 episodes and store final weights. The average score goes from 19 to 35 in very few steps and finally stabilizes in around 38.

![alt text](https://github.com/pablobd/ContinuousControl-D4PG/blob/master/d4pg_performance.PNG)

The reason to do this is that training was very time and resource consuming and a good strategy was to split the training in three stages. The agent configuration for the three stages remains identical. However, the buffer is cleaned at the beginning of the three stages and experience starts to be stored from scratch.


## Improvements

In the next release, we plan to add the following improvements:
* Batch normalization after each layer
* Prioritized learning to speed up training

