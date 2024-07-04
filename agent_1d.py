import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model_1d import Linear_QNet, QTrainer
from helper import plot_everything
import os
from enum import Enum
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Train_mode(Enum):
    """Enum train mode"""
    
    input_values_11 = 1
    input_values_19 = 2
    weight_loading = 3
    
# training mode - it switches experiments
train_mode = Train_mode.input_values_11

# rewrite best small model
rewrite_small_model = False

# maximum size of replay memory
# The aim of replay memory is to reduce correlation between consequtive steps in short memory
MAX_REPLAY_MEMORY = 100_000

# batch size for learning
BATCH_SIZE = 32

# learning rate
LR = 0.001

# loading state dictionary from a previous generation
small_model_state_dict = torch.load("model/model_11_values.pth")

# model with small input vector
small_model = Linear_QNet(11, 256, 3)

# model with big input vector for loading weights
big_model_for_uploading = Linear_QNet(19, 256, 3)

# model with big input vector for standalone training
big_model_standalone = Linear_QNet(19, 256, 3)

# (256,11)
input_weights_small_model = small_model_state_dict['linear_1.weight']

# creating a zero tensor
additional_input_part = torch.zeros(256, 8)

# concatinating with the first layer weights
evlolved_layer = torch.cat((input_weights_small_model, additional_input_part), dim=1)

# setting the newly formed layer - it's not already a small model state dict!
small_model_state_dict['linear_1.weight'] = evlolved_layer

# !!! Moment of truth - loading weights !!!
big_model_for_uploading.load_state_dict(small_model_state_dict)

class Agent:
    """Agent class"""

    def __init__(self):
        
        # game counter
        self.game_counter = 0
        
        self.epsilon = 0 # randomness
        
        # discount factor for Belman equation
        self.gamma = 0.9 # discount rate
        
        # replay memory
        self.replay_memory = deque(maxlen=MAX_REPLAY_MEMORY)
        
        # model
        self.model = None
        
        # choosing a model according to the experiment
        if train_mode == Train_mode.input_values_11:
            self.model = small_model
        elif train_mode == Train_mode.input_values_19:
            self.model = big_model_standalone
        elif train_mode == Train_mode.weight_loading:
            self.model = big_model_for_uploading
        else:
            raise("The train mode is not specified!")
        
        # that element that implements Belman equation
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
#-------------------------------------------------------------------------------
    def get_state(self, game):
        """Get state - 11 booleans based on the game"""
        
        # Here we return a list of parameters based on the game
        # 11 boolean variables - small state
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
        
        state_small = [
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
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]
        
        state_big = [
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
            
            # myself to the right
            game.myself_to_the_right_of_me,
            
            # myself to the left
            game.myself_to_the_left_of_me,
            
            # myself in front of me
            game.myself_to_the_front_of_me,
            
            game.front_wall,
            game.right_wall,
            game.left_wall,
            
            game.last_turn_right,
            game.last_turn_left
            ]
        
        # choosing the resultant state
        resultant_state = None
        
        if train_mode == Train_mode.input_values_11:
            resultant_state = state_small
        else:
            resultant_state = state_big

        return np.array(resultant_state, dtype=int)
#-------------------------------------------------------------------------------
    def remember_in_replay_memory(self, state, action, reward, next_state, done):
        """Remember in replay memory"""
        
        # just appending everything in a messy way
        # the ammount of states, actions, rewards - is determined by how long we played the game
        self.replay_memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
#-------------------------------------------------------------------------------
    def train_on_replay_memory(self):
        """Train on replay memory"""
        
        # if can collect a batch - get a random sample form a replay memory size of a batch, else - get the entire memory
        if len(self.replay_memory) > BATCH_SIZE:
            mini_sample = random.sample(self.replay_memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.replay_memory
        
        # an interesting function - returns a set of tuples
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # performing a train iteration with 
        self.trainer.train_step(states, actions, rewards, next_states, dones)
#-------------------------------------------------------------------------------
    def train_short_memory(self, state, action, reward, next_state, done):
        """Train a single iteration"""
        
        self.trainer.train_step(state, action, reward, next_state, done)
#-------------------------------------------------------------------------------
    def get_action(self, state):
        """Get an action from a model based on a state"""
        
        # random moves: tradeoff exploration / exploitation
        # defining epsilon 
        self.epsilon = 80 - self.game_counter
        
        returned_move = [0,0,0]
        
        # if we start from a newly trained model and we should use EXPLORATION
        if not train_mode == Train_mode.weight_loading and random.randint(0, 200) < self.epsilon: # 
            # getting the number of move where to set 1
            move = random.randint(0, 2)
            returned_move[move] = 1
        
        # use EXPLOITATION
        else:
            state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state)
            
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
    
    # creating an agent
    agent = Agent()
    
    # creating a game
    game = SnakeGameAI()
    
    # endless loop
    while True:
        
        # getting a current state from a game
        current_state = agent.get_state(game)

        # get an action from a model
        next_move = agent.get_action(current_state)
        
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
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(current_state, next_move, reward, state_new, done)

        # remember
        agent.remember_in_replay_memory(current_state, next_move, reward, state_new, done)

        # if we finished the game
        if done:
            
            # train long memory, plot result
            game.reset()
            
            # incrementing the game counter
            agent.game_counter += 1
            
            # train long memory after finishing the game
            # !!! replay memory or experience replay !!!
            # it trains against on all previous moves and games it's played and that helps to imporve
            agent.train_on_replay_memory()

            # if we reached the better score being in the state of training a small model
            if score > record_score:
                
                # saving the best record score
                record_score = score
                
                # if we should also rewrite
                if train_mode == Train_mode.input_values_11 and rewrite_small_model:
                    agent.model.save('model_11_values.pth')
            
            # unfreezing the second layer
            if(agent.game_counter == 50):
                big_model_for_uploading.linear_2.requires_grad_(True)
            
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

# if we start this file - start the training process
if __name__ == '__main__':
    train()