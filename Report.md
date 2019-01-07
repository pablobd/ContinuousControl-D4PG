# Report

## Solution

The agent is build following the D4PG [paper]((https://arxiv.org/abs/1509.02971)).

The agent learns actions with and Actor network with a single hidden layer of 264 units and estimates the score with a critic network with hidden layers of 128, 128 and 64 units.

The procedure works as follows:
* each agent adds its experience to a replay buffer that is shared by all agents, and
* the (local) actor and critic networks are updated 20 times in a row (one for each agent), using 20 different samples from the replay buffer.
* the (target) actor and critic networks are updated once every two updates.
* gradient clipping is used on the actor and critic networks

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
