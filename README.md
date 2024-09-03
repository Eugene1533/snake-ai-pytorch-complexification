## Overview
Training a relatively big neural network that has enough capacity for complex tasks is challenging. In real life the process of task solving requires system of knowledge, where more complex skills are built upon previously learned ones. The same way biological evolution builds new forms of life based on a previously achieved level of complexity. Inspired by that, this work proposes a way of training neural networks with smaller receptive fields and using their weights as prior knowledge for more complex successors through gradual involvement of some parts. That allows better performance in a particular case of deep Q-learning in comparison with a situation when the model tries to use a complex receptive field from scratch.

## Related work
Training a model on examples of increasing difficulty, progressively providing more challenging data or tasks as the policy improves, is called _Curriculum Learning_. As the name suggests, the idea behind the approach borrows from human education, where complex tasks are taught by breaking them into simpler parts. Another related approach is called _Progressive Neural Networks_. A progressive network is composed of multiple columns, and each column is a policy network for one specific task. It starts with one single column for training the first task, and then the number of columns increases with the number of new tasks. While training on a new task, neuron weights of the previous columns are frozen and representations from those frozen tasks are applied to the new column via a collateral connection to assist in learning the new task. Also the idea of _Distillation Model_ involves training a smaller model first and then building a big one that will imitate the first one in order to kick start the large model’s learning progress. In spite of some similarities, the suggested approach unlike others uses successive allocation of the network capacity for a current single task through increasing the perception field, i.e. the state space.

## Complexification through weight loading
These experiments are based on the original project of Snake game: https://github.com/patrickloeber/snake-ai-pytorch.
In order to reproduce the experiments run a file agent_1d.py. Enum Train_mode defines 3 possible scenarios. The mode input_values_11 involves feeding the snake with an input vector of 11 parameters that are relative to its head’s position. It takes approximately 100 games to converge and the average result is about 35 scores:

<p align="center">
  <img src="https://github.com/Eugene1533/snake-ai-pytorch-complexification/assets/174684744/0bc5cb9c-3909-415f-be99-b1ad4ac921ad" width="450"/>
</p>

The analysis of the way the snake ends up shows that it tends to coil in itself:

<p align="center">
  <img src="https://github.com/Eugene1533/snake-ai-pytorch-complexification/assets/174684744/63912fd4-63bc-4e33-ac39-6c57a6c4335d" width="350"/>
</p>

That situation is supposedly attributable to the inability of the snake to get understanding of the location of its own parts. If we choose input_values_19 mode, we increase the input vector by adding the following parameters:
-	snake tail to the right of the head;
-	snake tail to the left of the head;
-	snake tail to the front of the head;
-	relative distance to the right wall;
-	relative distance to the left wall;
-	relative distance to the front wall;
-	last turn left;
-	last turn right.

Now it takes approximately 150 games to converge and the average result is about 62:

<p align="center">
  <img src="https://github.com/Eugene1533/snake-ai-pytorch-complexification/assets/174684744/1035be08-61fc-43d5-93f2-ee0a4780b486" width="450"/>
</p>

If we want to start the learning process of a model with a bigger input vector not from scratch, but with weights of a smaller one, the procedure in this case is straightforward. For the current network architecture it requires copying of the weights of the second fully connected layer (FC 2) and the weights of the first fully connected layer (FC 1) concatenated with a tensor of random values of shape (8, 256) in order to fit the new formed FC 2 layer. The number of input and output features of each layer is specified in parenthesis:

<p align="center">
  <img src="https://github.com/Eugene1533/snake-ai-pytorch-complexification/assets/174684744/0fd87eeb-8298-4363-b172-6317557d1d75" width="670"/>
</p>

It corresponds to weight_loading mode and demonstrates that with prior knowledge it takes approximately 50 games to converge to even better scores in comparison with the experiment of a previous mode, which is about 3 times less in terms of count of games:

<p align="center">
  <img src="https://github.com/Eugene1533/snake-ai-pytorch-complexification/assets/174684744/b9547828-0f19-4d83-89e6-295245118a6d" width="450"/>
</p>

Due to the nature of neural networks, they can rely only on known part of the input vector, performing rational activity in terms of the environment and simultaneously figuring out the way of applying newly added part of the input vector. It’s important to note that it doesn’t require any initial exploration as it were in both cases with a smaller and bigger vectors starting from scratch. But it seems that in more complicated scenarios a way of exploring possibilities that come with added input vector might be required.

## Complexification through head involvement
The next step is to add a convolutional (2d) head to the neural network that will partially observe the environment – file agent_1d_2d.py. For this case a bit different approach will be demonstrated, which involves turning of some advanced parts of the neural network, like the convolutional head in this example, while training the initial smaller parts. In essence, this process is similar to training a smaller neural network and loading its weights into a correspondent part of a bigger one. In order to make it easier for the agent to learn, the convolutional head is provided not with the full environment, but with black and white cropped fragment of shape (8, 8) around snake’s head, rotated according to its current direction:

<p align="center">
  <img src="https://github.com/Eugene1533/snake-ai-pytorch-complexification/assets/174684744/7b72cb2b-2873-4788-9940-317acb0730cc" width="570"/>
</p>

Architecture of the neural network (without ReLU layers) is presented on consists of two heads:

<p align="center">
  <img src="https://github.com/user-attachments/assets/c3f5a343-bd5f-417d-8965-97cecb22a66b" width="550"/>
</p>

There are several sequential stages of training. During a “Zeros” stage the output of the convolutional head is always a tensor of zeros and the head is frozen. In this case the agent is supposed to rely only on the 1d head. A “Noise” stage involves processing the image by the frozen convolutional head with randomly initialized weights. The absence of any structured useful information about the environment from 2d head supposedly will make the rest of the network insensitive to any information from that head. The initial intent of that is to prevent possible sporadic behavior of the network on the transition between the previous stage and involving the 2d head, when the network has been trained with the constant tensor of only zeros and it unexpectedly gets a tensor of random values. An “Involving” stage implies freezing the 1d head and unfreezing the 2d head in order to provide some prior knowledge and kick start the learning process of 2d head. A “Both heads” stage involves simultaneous training of both 1d and 2d heads.
	A set of experiments has been conducted in order to practically evaluate performance, depending on redistribution of the entire amount of 3000 games between different stages using fixed hyperparameters. Every experiment the agent uses epsilon-greedy strategy during first 280 games and then the greedy one. The first experiment involves training the agent during all of the episodes (games) using a “Zeros” stage, which means it effectively uses only 1d head (“just_1d” configuration in the code):

<p align="center">
  <img src="https://github.com/Eugene1533/snake-ai-pytorch-complexification/assets/174684744/875a3e9b-06c2-4a4b-9fbb-27df34a3c88c" width="450"/>
</p>

Unsurprisingly, the result doesn’t seem much different from Fig. 2. It has the average score of 33 over 100 last games. The network just learns to ignore a tensor of zeros from the 2d head and rely only on 1d part that uses 11 values, the same as in case on Figure 2. It’s necessary to mention that in spite of pretty stable average score, the dispersion of scores for each game (blue color) is pretty high.
The second experiment involves training the agent during all of the episodes using the “Both heads” stage, which means it uses both heads from the beginning (“combined” configuration in the code):

<p align="center">
  <img src="https://github.com/Eugene1533/snake-ai-pytorch-complexification/assets/174684744/224661cb-b825-4bb0-9c29-ca339b690e6c" width="450"/>
</p>

It has the average score of 36 over 100 last games. The result is not far from the previous experiment, which means that the network isn’t able to utilize data from the 2d head, “turned that head off”, and still relies only on 1d head as in the case with “Zeros” stage.
The third experiment involves training the agent on 500 games using “Zeros” stage and 2500 games using “Both heads” stage (“1d+2d” configuration in the code):

<p align="center">
  <img src="https://github.com/Eugene1533/snake-ai-pytorch-complexification/assets/174684744/60864637-9297-4144-994f-699f92398bba" width="450"/>
</p>

In this case during the first stage the network learns how to utilize the 1d head and then, with its weights trained, involves the second one in the training process. The average score over 100 last games is 54, which is better than in the previous cases. The important point here is that such a score can’t be achieved by training of two heads simultaneously. The forth experiment involves training the agent on 500 games using “Zeros” stage, 1000 games using “Involving” stage, and 1500 games using “Both heads” stage (“1d+involving+2d” configuration in the code):

<p align="center">
  <img src="https://github.com/Eugene1533/snake-ai-pytorch-complexification/assets/174684744/fe08d442-0ab6-47e5-9cfa-b411f64c3c65" width="450"/>
</p>

The final average score is 54 and is the same as in the previous experiment, which means that using “Involving” stage doesn’t improve the results. The fifth experiment involves training the agent on 500 games using “Zeros” stage, 500 games using “Noise” stage, 500 games using “Involving” stage, and 1500 games using “Both heads” stage (“1d+noise+involving+2d” configuration in the code):

<p align="center">
  <img src="https://github.com/Eugene1533/snake-ai-pytorch-complexification/assets/174684744/db3e0a49-ff3c-44dc-a9d1-6766bb7b0122" width="450"/>
</p>

Here also the final average score of 52 doesn’t deviate too much from the third experiment, which means that using “Involving” stage also doesn’t improve the results.

## Complexification through reward shaping
Transition from a simpler neural network to a more complicated one can also be conducted through reward shaping. The set of corresponding experiments is presented in the file agent_reward_shaping.py. In the first experiment (second_level_network_from_scratch mode) the network is provided with the entire game screen rotated relatively to its head:

<p align="center">
  <img src="https://github.com/user-attachments/assets/32a3af7e-32f1-42c0-94da-abae02f6071a" width="370"/>
</p>

The network has the following architecture:

<p align="center">
  <img src="https://github.com/user-attachments/assets/c23feaf2-a50c-4cf2-97b4-f4dd5de2b167" width="450"/>
</p>

And it demonstrates the following result using epsilon-greedy strategy for exploration during first 5000 games:

<p align="center">
  <img src="https://github.com/user-attachments/assets/511c4ec1-f18a-4e3e-b61a-d8810969933b" width="450"/>
</p>

As it’s shown, with the established set of hyperparameters, the network doesn’t converge at all. At the same time, the approach that has been described earlier, of using liner layer with the manually constructed 11 values demonstrates the following result in this case (primodial_network_training mode), with an average score about 40:

<p align="center">
  <img src="https://github.com/user-attachments/assets/dae402ba-4a01-4d9b-9d80-a6b507eaa2f2" width="450"/>
</p>

The suggested mechanism of complexification through reward shaping involves usage of predicted value function from a smaller neural network trained during the first stage in the same environment as a part of reward for a more complicated one:

<p align="center">
  <img src="https://github.com/user-attachments/assets/1adf388a-b86b-4543-a563-f18c9a4899a2" width="450"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/d7980b2c-c36b-47ba-978c-ec0c020826ec" width="650"/>
</p>

The result of training when reward is a maximum value of a Q-function for a given state form a smaller network is the following (second_level_network_reward_shaping mode):

<p align="center">
  <img src="https://github.com/user-attachments/assets/a3695d45-9d68-425f-aba0-9d440148855d" width="450"/>
</p>

The result of training with an equal contribution of a real reward and a maximum Q-function for a given state form a smaller network is the following:

<p align="center">
  <img src="https://github.com/user-attachments/assets/cdb89c15-1ea4-4fe8-a0e0-defad5d21d4e" width="450"/>
</p>

The same experiment but with longer exploration phase looks like:

<p align="center">
  <img src="https://github.com/user-attachments/assets/c3d3426e-6ea0-4dce-9293-b4777dda7a4d" width="450"/>
</p>

It has an average score about 50 which is better in comparison with the case with just using 11 values. So the result of a bigger network trained using some reward function provided by the smaller network is better than the result of the smaller network in that environment, which is connected with richer state space in this particular case. And more importantly, the bigger network doesn’t converge at all without using such a gimmick. It also sees reasonable to shift the reward from a smaller network to the real one from the environment over time of training the bigger network and further research can be dedicated to that, as well as to using a chain of successively trained networks where each previous one provides reward construction for the next one.

## Complexification through assistance providing
In the previous case the bigger network acted in the environment but it was rewarded by a smaller one. It corresponds to a script: “You’ll act and I’ll tell you what’s good or bad”. For the purpose of research it seems reasonable to consider an alternative scenario which corresponds to: “You’ll be provided with some experiments by me, and the environment will tell you the outcome of certain actions”. During the second stage of training (agent_1d_assistance_provider.py) actual behavior is generated by a pretrained assistant model, and the agent is trained on the experience replay buffer, but instead of input vector of 11 parameters it uses the correspondent 2d representation:

<p align="center">
  <img src="https://github.com/user-attachments/assets/363873873-a4752f14-968f-44f3-97eb-8807986dec36" width="650"/>
</p>

The agent is switching between the aforementioned way of training and training on its own in order to assess its performance. The result of training is worse than the previous one:

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/174684744/363873900-832a3887-a3d0-4dbf-8093-eee60b73a623.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjUzNDc4MDgsIm5iZiI6MTcyNTM0NzUwOCwicGF0aCI6Ii8xNzQ2ODQ3NDQvMzYzODczOTAwLTgzMmEzODg3LWEzZDAtNGRiZi04MDkzLWVlZTYwYjczYTYyMy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwOTAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDkwM1QwNzExNDhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04NDVkOGNiNDY5MmZmOGMzOWMxZjA3Njg1MjAwMTA1MjEyNTJlMTgwZDAxZTZiNDM5MWI2OWExYTRjOTljZGZjJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.a35tM24ZznWwByItf3kVUvGBc_FbD2PV05QAYm7x6lk" width="450"/>
</p>


## Conclusions
This work is based on considering a pretty trivial example of the Snake game and describes the process of using weights of a previously trained network as prior knowledge for a more complicated one, as well the process of reward providing. It was shown that the suggested approaches provide a way of achieving higher scores without any hyperparamenter search in comparison with the cases of training complexified networks from scratch. Future work requires conducting a more extensive set of experiments, including different environments and RL algorithms for getting conclusive information about applicability of the approaches. It’s necessary to consider different possible dimensions of increasing complexity, not only what’s directly connected with a receptive filed, i.e. a state vector. It seems that in this particular case of the Snake game we can use not a single current state of the game, but also several previous states and gradually add some recurrent part to the network. Future research can also be dedicated to finding automatically the necessary directions of extending network capacity, unlike it was done manually in the current work. In case of reward provider it also seems reasonable to shift the reward from a smaller network to the real one from the environment over time of training and further research can be dedicated to that, as well as to use a chain of successively trained networks where each previous one provides reward construction for the next one. Usage of a combination of the reward providing and assistance providing techniques may also be studied in the future. A sense of feeling a distance of food, not just direction of that, may also improve results. In this case it requires a gradual transition from a Boolean value to a value between 0 and 1.
