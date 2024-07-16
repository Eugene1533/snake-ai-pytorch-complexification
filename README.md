## Overview
Training a relatively big neural network that has enough capacity for complex tasks is challenging. In real life the process of task solving requires system of knowledge, where more complex skills are built upon previously learned ones. The same way biological evolution builds new forms of life based on a previously achieved level of complexity. Inspired by that, this work proposes a way of training neural networks with smaller receptive fields and using their weights as prior knowledge for more complex successors through gradual involvement of some parts. That allows better performance in a particular case of deep Q-learning in comparison with a situation when the model tries to use a complex receptive field from scratch.

## Related work
Training a model on examples of increasing difficulty, progressively providing more challenging data or tasks as the policy improves, is called _Curriculum Learning_. As the name suggests, the idea behind the approach borrows from human education, where complex tasks are taught by breaking them into simpler parts. Another related approach is called _Progressive Neural Networks_. A progressive network is composed of multiple columns, and each column is a policy network for one specific task. It starts with one single column for training the first task, and then the number of columns increases with the number of new tasks. While training on a new task, neuron weights of the previous columns are frozen and representations from those frozen tasks are applied to the new column via a collateral connection to assist in learning the new task. Also the idea of _Distillation Model_ involves training a smaller model first and then building a big one that will imitate the first one in order to kick start the large model’s learning progress. In spite of some similarities, the suggested approach unlike others uses successive allocation of the network capacity for a current single task through increasing the perception field, i.e. the state space.

## Complexification through weight loading
In order to reproduce the experiments run a file agent_1d.py. Enum Train_mode defines 3 possible scenarios. The mode input_values_11 involves feeding the snake with an input vector of 11 parameters that are relative to its head’s position. It takes approximately 100 games to converge and the average result is about 35 scores.





