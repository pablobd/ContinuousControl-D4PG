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
* the Critic network with 33 input units (state size), hidden layers of 256, 128 + 4 (action size), 128 and 64 units, and an output layer of 1 unit,
* the Actor network has an input layer of 33 units (state size), a single hidden layer of 264 units, and an output layer of 4 units (action size).

The following values ofr hyperparameters are used:
* BATCH_SIZE = 128
* GAMMA = 0.99
* TAU = 1e-3
* LR_ACTOR = 1e-4
* LR_CRITIC = 3e-4
* WEIGHT_DECAY_CR = 0.0001
* WEIGHT_DECAY_AC = 0

## Results

This solution reaches a score of 22 in 50 periods but fails to continue learning and stabilizes around this number.

![alt text](https://github.com/pablobd/ContinuousControl-D4PG/blob/master/d4pg_performance.PNG)


## Improvements

In the next release, we want to add prioritized learning to speed up training.
