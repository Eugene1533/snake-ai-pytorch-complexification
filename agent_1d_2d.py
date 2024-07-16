import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, train_stage
from model_1d_2d import QTrainer, Composite_Net
from helper import plot_everything
from statistics import mean
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# maximum size of replay memory
# The aim of replay memory is to reduce correlation between consequtive steps in short memory
MAX_REPLAY_MEMORY = 5000

# batch size for learning
BATCH_SIZE = 32

# should we load a previous model
start_from_previous_model = False

# must we use extended 1d vector
must_use_extended_1d = False

# learning rate
LR = 0.001

# creating named windows
cv2.namedWindow('Snake', cv2.WINDOW_NORMAL)
cv2.namedWindow('Snake rotated', cv2.WINDOW_NORMAL)
cv2.namedWindow('Big', cv2.WINDOW_NORMAL)
cv2.namedWindow('Head crop', cv2.WINDOW_NORMAL)

# a composite network consists of two heads
composite_net = Composite_Net(512)

# set of experiments
experiments_configurations = {}

# 1's in case of an empty stage need for correct work and don't effect the result
experiment_configuration_1 = {train_stage.Zeros_1 : (3000, 0.0005), train_stage.Noise_2 : (1, 0.0005), train_stage.Involving_3 : (1, 0.0005), train_stage.Both_heads_4 : (1, 0.0005)}
experiments_configurations['just_1d'] = experiment_configuration_1

experiment_configuration_2 = {train_stage.Zeros_1 : (1, 0.0005), train_stage.Noise_2 : (1, 0.0005), train_stage.Involving_3 : (1, 0.0005), train_stage.Both_heads_4 : (3000, 0.0005)}
experiments_configurations['combined'] = experiment_configuration_2

experiment_configuration_3 = {train_stage.Zeros_1 : (500, 0.0005), train_stage.Noise_2 : (1, 0.0005), train_stage.Involving_3 : (1, 0.0005), train_stage.Both_heads_4 : (2500, 0.0005)}
experiments_configurations['1d+2d'] = experiment_configuration_3

experiment_configuration_4 = {train_stage.Zeros_1 : (500, 0.0005), train_stage.Noise_2 : (1, 0.0005), train_stage.Involving_3 : (1000, 0.0005), train_stage.Both_heads_4 : (1500, 0.0005)}
experiments_configurations['1d+involving+2d'] = experiment_configuration_4

experiment_configuration_5 = {train_stage.Zeros_1 : (500, 0.0005), train_stage.Noise_2 : (500, 0.0005), train_stage.Involving_3 : (500, 0.0005), train_stage.Both_heads_4 : (1500, 0.0005)}
experiments_configurations['1d+noise+involving+2d'] = experiment_configuration_5

# filling the list of experiments
list_of_experiment_names = []
for key, value in experiments_configurations.items():
    list_of_experiment_names.append(key)

list_of_stages_in_experiment = []

class Agent:
    """Agent class"""

    def __init__(self, game):
        
        # game counter
        self.game_counter = 0
        
        # game counter
        self.game_counter_of_training_stage = 0
        
        # injecting the game into an agent
        self.game = game
        
        # randomness
        self.epsilon = 0
        
        # discount factor for Belman equation
        self.gamma = 0.9 # discount rate
        
        # replay memory
        self.replay_memory = deque(maxlen=MAX_REPLAY_MEMORY)
        
        # model
        self.model = composite_net
        self.model.to(device)
        
        # that element that implements Belman equation
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # dictionary of new states that indicates their fact of visiting
        self.dic_of_new_states = dict()
        
        # initializing the 2d state, in the future it will be changed
        self.state_2d = np.zeros((int(self.game.h / 20) + 1, int(self.game.w / 20) + 1))
#-------------------------------------------------------------------------------
    def get_state(self, game):
        """Get state - 11 booleans based on the game"""
        
        # Here we return a list of parameters based on the game
        # 11 boolean variables !!!
        head = game.snake[0]
        point_left = Point(head.x - 20, head.y)
        point_right = Point(head.x + 20, head.y)
        point_up = Point(head.x, head.y - 20)
        point_down = Point(head.x, head.y + 20)
        
        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN
        
        # elemetns to the right of the head in absolute coordinates
        elements_to_the_right_abs = any(map(lambda pt: pt.x > head.x and pt.y == head.y, game.snake))
        
        # elemetns to the left of the head in absolute coordinates
        elements_to_the_left_abs = any(map(lambda pt: pt.x < head.x and pt.y == head.y, game.snake))
        
        # elemetns to the top of the head in absolute coordinates
        elements_to_the_top_abs = any(map(lambda pt: pt.y < head.y and pt.x == head.x, game.snake))
        
        # elemetns to the bottom of the head in absolute coordinates
        elements_to_the_bottom_abs = any(map(lambda pt: pt.y > head.y and pt.x == head.x, game.snake))
        
        game.myself_to_the_right_of_me = False
        game.myself_to_the_left_of_me = False
        game.myself_to_the_front_of_me = False
        
        # if we're going to the right
        if(game.direction == Direction.RIGHT):
            # if I have something to the top of me
            if(elements_to_the_top_abs):
                game.myself_to_the_left_of_me = True
            if(elements_to_the_bottom_abs):
                game.myself_to_the_right_of_me = True
            if(elements_to_the_right_abs):
                game.myself_to_the_front_of_me = True
        # if we're going to the left
        elif(game.direction == Direction.LEFT):
            # if I have something to the top of me
            if(elements_to_the_top_abs):
                game.myself_to_the_right_of_me = True
            if(elements_to_the_bottom_abs):
                game.myself_to_the_left_of_me = True
            if(elements_to_the_left_abs):
                game.myself_to_the_front_of_me = True
        # if we're going up
        elif(game.direction == Direction.UP):
            # if I have something to the right of me
            if(elements_to_the_right_abs):
                game.myself_to_the_right_of_me = True
            if(elements_to_the_left_abs):
                game.myself_to_the_left_of_me = True
            if(elements_to_the_top_abs):
                game.myself_to_the_front_of_me = True
        # if we're going down
        elif(game.direction == Direction.DOWN):
            # if I have something to the right of me
            if(elements_to_the_right_abs):
                game.myself_to_the_left_of_me = True
            if(elements_to_the_left_abs):
                game.myself_to_the_right_of_me = True
            if(elements_to_the_bottom_abs):
                game.myself_to_the_front_of_me = True

        # distances to the walls
        right_wall_abs = (game.w - head.x)/game.w
        left_wall_abs = head.x/game.w
        up_wall_abs = head.y/game.h
        down_wall_abs = (game.h - head.y)/game.h
        
        game.front_wall = 0
        game.right_wall = 0
        game.left_wall = 0
        
        # if we're moving right
        if(dir_right):
            game.front_wall = right_wall_abs
            game.right_wall = down_wall_abs
            game.left_wall = up_wall_abs
        elif(dir_left):
            game.front_wall = left_wall_abs
            game.right_wall = up_wall_abs
            game.left_wall = down_wall_abs
        elif(dir_up):
            game.front_wall = up_wall_abs
            game.right_wall = right_wall_abs
            game.left_wall = left_wall_abs
        elif(dir_down):
            game.front_wall = down_wall_abs
            game.right_wall = left_wall_abs
            game.left_wall = right_wall_abs
            
        state = [
            # Danger straight
            (dir_right and game.is_collision(point_right)) or 
            (dir_left and game.is_collision(point_left)) or 
            (dir_up and game.is_collision(point_up)) or 
            (dir_down and game.is_collision(point_down)),

            # Danger right
            (dir_up and game.is_collision(point_right)) or 
            (dir_down and game.is_collision(point_left)) or 
            (dir_left and game.is_collision(point_up)) or 
            (dir_right and game.is_collision(point_down)),

            # Danger left
            (dir_down and game.is_collision(point_right)) or 
            (dir_up and game.is_collision(point_left)) or 
            (dir_right and game.is_collision(point_up)) or 
            (dir_left and game.is_collision(point_down)),
            
            # Move direction
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            
            # # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
            ]
        
        return np.array(state, dtype=int)
    #-------------------------------------------------------------------------------
    def get_2d_state(self, game):
        
        # clearing a state array
        self.state_2d = np.full((33,33), 0.1)
        
        big_2d_field = np.zeros((132,132))
        
        head = game.snake[0]
        food = game.food
        
        # going through all of the snake
        for point in game.snake:
            
            if point.x == 0:
                print()
                
            if point.y == 0:
                print()
            
            
            x = int(point.x/20)
            y = int(point.y/20)
            
            self.state_2d[y][x] = 0.25
        
        # setting the head
        self.state_2d[int(head.y/20)][int(head.x/20)] = 0.5
        
        # setting the food
        self.state_2d[int(food.y/20)][int(food.x/20)] = 1
        
        cv2.imshow('Snake', self.state_2d)

        tensor_state_2d_rotated = torch.tensor(self.state_2d, dtype=torch.float)

        # rotating current state accroding to the orintation of the snake
        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN
        
        # if we're moving up - we dont rotate anything
        if dir_up:
            tensor_state_2d_rotated = tensor_state_2d_rotated
        elif dir_right:
            tensor_state_2d_rotated = torch.rot90(tensor_state_2d_rotated, k=1)
        elif dir_left:
            tensor_state_2d_rotated = torch.rot90(tensor_state_2d_rotated, k=-1)
        elif dir_down:
            tensor_state_2d_rotated = torch.rot90(tensor_state_2d_rotated, k=2)
        
        # converting a tensor to numpy 
        tensor_state_2d_rotated_numpy = tensor_state_2d_rotated.numpy()
        
        cv2.imshow('Snake rotated', tensor_state_2d_rotated_numpy)
        
        first_dim_left = int(self.state_2d.shape[0] + self.state_2d.shape[0]/2)
        first_dim_right = int(self.state_2d.shape[0]/2) + 2 * self.state_2d.shape[0]
        
        second_dim_left = int(self.state_2d.shape[1] + self.state_2d.shape[1]/2)
        second_dim_right = int(self.state_2d.shape[1]/2) + 2 * self.state_2d.shape[1]
        
        # replacing the center of a big array with a target one
        big_2d_field[first_dim_left:first_dim_right, second_dim_left:second_dim_right] = tensor_state_2d_rotated_numpy
        cv2.imshow('Big', big_2d_field)
        
        # getting the crop relative to head
        # getting relative coordinates of the head when an image is rotated
        head_coordinates = np.argwhere(big_2d_field == 0.5)
        
        head_x = head_coordinates[0][0]
        head_y = head_coordinates[0][1]
        
        head_relative_crop = big_2d_field[head_x - int(self.state_2d.shape[0]/8) : head_x + int(self.state_2d.shape[0]/8), head_y - int(self.state_2d.shape[1]/8) : head_y + int(self.state_2d.shape[1]/8)]
        
        cv2.imshow('Head crop', head_relative_crop)
        
        return head_relative_crop
#-------------------------------------------------------------------------------
    def remember_in_replay_memory(self, current_state_1d, current_state_2d, next_move, reward, state_new_1d, state_new_2d, done):
        """Remember in replay memory"""
        
        # just appending everything in a messy way
        # the ammount of states, actions, rewards - is determined by how long we played the game
        self.replay_memory.append((current_state_1d, current_state_2d, next_move, reward, state_new_1d, state_new_2d, done)) # popleft if MAX_MEMORY is reached
#-------------------------------------------------------------------------------
    def train_on_replay_memory(self):
        """Train on replay memory"""
        
        # if can collect a batch - get a random sample form a replay memory size of a batch, else - get the entire memory
        if len(self.replay_memory) > BATCH_SIZE:
            mini_sample = random.sample(self.replay_memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.replay_memory
        
        # an interesting function - returns a set of tuples
        current_state_1d, current_state_2d, next_move, reward, state_new_1d, state_new_2d, done = zip(*mini_sample)
        
        # performing a train iteration with
        self.trainer.train_step(current_state_1d, current_state_2d, next_move, reward, state_new_1d, state_new_2d, done)
#-------------------------------------------------------------------------------
    def train_short_memory(self, current_state_1d, current_state_2d, next_move, reward, state_new_1d, state_new_2d, done):
        """Train a single iteration"""
        
        self.trainer.train_step(current_state_1d, current_state_2d, next_move, reward, state_new_1d, state_new_2d, done)
#-------------------------------------------------------------------------------
    def get_action(self, state_1d, state_2d, game):
        """Get an action from a model based on a state"""
        
        # random moves: tradeoff exploration / exploitation
        border_game_exploration_number = 280
        
        self.epsilon = border_game_exploration_number - self.game_counter
        
        returned_move = [0,0,0]
        
        # if we start from a newly trained model and we should use EXPLORATION
        if not start_from_previous_model and random.randint(0, 600) < self.epsilon: # 
            # getting the number of move where to set 1
            move = random.randint(0, 2)
            returned_move[move] = 1
                
        # use EXPLOITATION
        else:
            state_1d = torch.tensor(state_1d, dtype=torch.float)
            state_1d = state_1d.to(device)
            state_1d = torch.unsqueeze(state_1d, 0)
            state_1d = torch.unsqueeze(state_1d, 0)
            
            state_2d = torch.tensor(state_2d, dtype=torch.float)
            state_2d = state_2d.to(device)
            state_2d = torch.unsqueeze(state_2d, 0)
            state_2d = torch.unsqueeze(state_2d, 0)
            
            prediction = self.model(state_1d, state_2d)
            
            # taking an action with the maximum probability
            move = torch.argmax(prediction).item()
            
            # setting one into that number
            returned_move[move] = 1

        return returned_move
#-------------------------------------------------------------------------------
def train():
    """Train the model - entery point of the project"""
    
    # list of scores for plotting
    scores_for_plotting = []
    
    # list of mean scores for plotting
    mean_scores_for_plotting = []
    
    # sum of scores
    sum_of_scores = 0
    
    # best score
    record_score = 0
    
    # creating a game
    game = SnakeGameAI()
    
    # creating an agent
    agent = Agent(game)
    
    index_of_experiment = 0
    
    # getting the name of an experiment and a correspondent list of stages
    name_of_experiment = list_of_experiment_names[index_of_experiment]
    train_configuration_current = experiments_configurations[name_of_experiment]
    
    # getting the stage list
    stage_list=list(train_configuration_current)
    
    # loading the model - if we start form the previous one
    if(start_from_previous_model):
        agent.model.load()
        agent.model.train()
        
    # setting the initial mode
    agent.model.set_training_stage(train_stage.Zeros_1)
    
    # endless loop
    while True:
        
        current_state_2d = agent.get_2d_state(game)
        # current_state_2d = current_state_2d.to(device)
        
        current_state_1d = agent.get_state(game)
        # current_state_1d = current_state_1d.to(device)

        # get an action from a model
        next_move = agent.get_action(current_state_1d, current_state_2d, game)
        
        # right turn
        if (np.array_equal(next_move, [0, 1, 0])):
            game.last_turn_right = True
            game.last_turn_left = False
        # left turn
        elif(np.array_equal(next_move, [0, 0, 1])):
            game.last_turn_right = False
            game.last_turn_left = True

        # perform an action and get new state
        reward, done, score = game.play_iteration(next_move)
		
        state_new_2d = agent.get_2d_state(game) 
        state_new_1d = agent.get_state(game)
        
        state_new_1d = torch.tensor(state_new_1d, dtype=torch.float)
        state_new_1d = torch.unsqueeze(state_new_1d, dim= 0)
        
        #agent.model.train()
        agent.train_short_memory(current_state_1d, current_state_2d, next_move, reward, state_new_1d, state_new_2d, done)
        
        # remember
        #agent.model.train()
        agent.remember_in_replay_memory(current_state_1d, current_state_2d, next_move, reward, state_new_1d, state_new_2d, done)

        # if we finished the game
        if done:
            
            # train long memory, plot result
            game.reset()
            
            # incrementing the game counter
            agent.game_counter += 1
            agent.game_counter_of_training_stage +=1
            
            # train long memory after finishing the game
            # !!! replay memory or experience replay !!!
            # it trains against on all previous moves and games it's played and that helps to imporve
            agent.train_on_replay_memory()

            # if we reached the better score - save the network
            if score > record_score:
                record_score = score
                agent.model.save('support_network_best.pth')
            
            # printing the information
            print('Game', agent.game_counter, 'Score', score, 'Record:', record_score)

            # appending the score for plotting
            scores_for_plotting.append(score)
            
            # getting the mean of scores
            # getting last 10 values
            last_values = []
            
            if len(scores_for_plotting) <10:
                last_values = scores_for_plotting
            else:
                last_values = scores_for_plotting[len(scores_for_plotting)-10:]
            
            sum_of_scores = sum(last_values)
            mean_score = sum_of_scores / len(last_values)
            mean_scores_for_plotting.append(mean_score)
            
            # plotting everything
            plot_everything(scores_for_plotting, mean_scores_for_plotting)
            
            # if we got till the end of the training period for the current stage
            if agent.game_counter_of_training_stage >= train_configuration_current[agent.model.get_training_stage()][0]:
            
                # if we have the next stages
                if agent.model.get_training_stage() != stage_list[len(stage_list)-1]:
                    
                    # getting the current index
                    current_state_index = stage_list.index(agent.model.get_training_stage())
                    
                    # setting the next stage with the index incremented
                    agent.model.set_training_stage(stage_list[current_state_index+1])
                    
                    # discarding the counter of the current stage
                    agent.game_counter_of_training_stage = 0
                    
                    # getting the learning rate
                    lr_internal = train_configuration_current[agent.model.get_training_stage()][1]
                    
                    # setting the lerning rate
                    agent.trainer.set_lr(lr_internal)
                    
                    # printing the name of the current stage
                    print("!!! set next trainng stage : ", agent.model.get_training_stage())
                    print("Learning rate : ", lr_internal)
                                
                # if we got till the end
                else:
                    
                    # save the current image of the graphic
                    plt.savefig('Images/'+name_of_experiment + '.png')
                    
                    # saving a trained model
                    agent.model.save(name_of_experiment +'.pth')

                    # creating a file for the report
                    f = open('Images/'+name_of_experiment + '.txt', "w")

                    # adding the maximum value
                    f.write('Maximum: ' + str(record_score) + '\n')

                    # getting the average of last 100 values
                    last_value = scores_for_plotting[len(scores_for_plotting)-100:]
                    average = mean(last_value)
                    
                    # adding the average value
                    f.write('Average: ' + str(average))
                    
                    # closing a file
                    f.close()

                    # discarding the record
                    record_score = 0
                    
                    # creating a new model - through creating a new agent
                    # replay buffer and other things will be reinstantiated
                    agent = Agent(game)

                    # setting the initial mode
                    agent.model.set_training_stage(train_stage.Zeros_1)
                    
                    # if we got to the last experiment
                    if index_of_experiment == len(list_of_experiment_names) - 1:
                        input("Set of games is over...")
                    
                    else:
                        # incrementing the index
                        index_of_experiment +=1
                        
                        # getting the name of an experiment and a correspondent list of stages
                        name_of_experiment = list_of_experiment_names[index_of_experiment]
                        train_configuration_current = experiments_configurations[name_of_experiment]
                        
                        # getting the stage list
                        stage_list.clear()
                        stage_list=list(train_configuration_current)
                        
                        # clearing the training data
                        agent.game_counter = 0
                        scores_for_plotting.clear()
                        mean_scores_for_plotting.clear()

# if we start this file - start the training process
if __name__ == '__main__':
    train()