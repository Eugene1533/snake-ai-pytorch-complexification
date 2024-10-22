import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model_1d import Linear_QNet, QTrainer
from model_2d import Model2d, QTrainer_2d
from helper import plot_everything
import os
from enum import Enum
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from itertools import islice
import cv2

class Train_mode(Enum):
    """Enum train mode"""
    
    primodial_network_training = 1
    second_level_network_from_scratch = 2
    second_level_network_reward_shaping = 3

# training mode
train_mode = Train_mode.second_level_network_reward_shaping

TargetTrainer = None
if train_mode == Train_mode.primodial_network_training:
    TargetTrainer = QTrainer
else:
    TargetTrainer = QTrainer_2d
    # creating named windows
    cv2.namedWindow('Snake', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Snake rotated', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Big', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Head crop', cv2.WINDOW_NORMAL)

# rewrite best small model
rewrite_small_model = True

# maximum size of replay memory
# The aim of replay memory is to reduce correlation between consequtive steps in short memory
MAX_REPLAY_MEMORY = 100_000

# batch size for learning
BATCH_SIZE = 32

# learning rate
LR = 0.0005

# loading state dictionary from a previous generation
small_model_state_dict = torch.load("model/model_11_values.pth")

# model with small input vector
primodial_model = Linear_QNet(11, 256, 3)

# model with a bigger input vector
second_level_model = Model2d(34848).to('cuda') # Linear_QNet(19, 256, 3)

# if we're in the second level model training mode
if train_mode == Train_mode.second_level_network_reward_shaping:
    primodial_model.load_state_dict(small_model_state_dict)

class Agent:
    """Agent class"""

    def __init__(self):
        
        # game counter
        self.game_counter = 0
        
        self.epsilon = 0 # randomness
        
        # discount factor for Belman equation
        self.gamma = 0.9 # discount rate
        self.gamma_cortizol = 0.5
        
        # replay memory
        self.replay_memory = deque(maxlen=MAX_REPLAY_MEMORY)
        
        # model
        self.model = None
        
        # choosing a model according to the training mode
        if train_mode == Train_mode.primodial_network_training:
            self.model = primodial_model
        elif train_mode == Train_mode.second_level_network_reward_shaping:
            self.model = second_level_model
        elif train_mode == Train_mode.second_level_network_from_scratch:
            self.model = second_level_model

        self.steps_without_real_future_reward = 0
        
        # that element that implements Belman equation
        self.trainer = TargetTrainer(self.model, lr=LR, gamma=self.gamma)
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
        resultant_state = state_small

        return np.array(resultant_state, dtype=int)
#-------------------------------------------------------------------------------
    def get_difficult_state(self, game):
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
        elements_to_the_right = [pt for pt in game.snake if pt.x > head.x and pt.y == head.y]
        elements_to_the_right_abs = 1
        if len(elements_to_the_right) != 0:
            elements_to_the_right_abs = (min(pt.x for pt in elements_to_the_right) - head.x)/game.w
        
        # elemetns to the left of the head in absolute coordinates
        elements_to_the_left = [pt for pt in game.snake if pt.x < head.x and pt.y == head.y]
        elements_to_the_left_abs = 1
        if len(elements_to_the_left) != 0:
            elements_to_the_left_abs = (head.x - max(pt.x for pt in elements_to_the_left))/game.w
        
        # elemetns to the top of the head in absolute coordinates
        elements_to_the_top = [pt for pt in game.snake if pt.y < head.y and pt.x == head.x]
        elements_to_the_top_abs = 1
        if len(elements_to_the_top) != 0:
            elements_to_the_top_abs = (min(pt.y for pt in elements_to_the_top) - head.y)/game.h
        
        # elemetns to the bottom of the head in absolute coordinates
        elements_to_the_bottom = [pt for pt in game.snake if pt.y > head.y and pt.x == head.x]
        elements_to_the_bottom_abs = 1
        if len(elements_to_the_bottom) != 0:
            elements_to_the_bottom_abs = (head.y - max(pt.y for pt in elements_to_the_bottom))/game.h
        
        game.myself_to_the_right_of_me = False
        game.myself_to_the_left_of_me = False
        game.myself_to_the_front_of_me = False
        
        # if we're going to the right
        if(game.direction == Direction.RIGHT):
            # if I have something to the top of me
            game.myself_to_the_left_of_me = elements_to_the_top_abs
            game.myself_to_the_right_of_me = elements_to_the_bottom_abs
            game.myself_to_the_front_of_me = elements_to_the_right_abs
        
        # if we're going to the left
        elif(game.direction == Direction.LEFT):
            # if I have something to the top of me
            game.myself_to_the_right_of_me = elements_to_the_top_abs
            game.myself_to_the_left_of_me = elements_to_the_bottom_abs
            game.myself_to_the_front_of_me = elements_to_the_left_abs
        
        # if we're going up
        elif(game.direction == Direction.UP):
            # if I have something to the right of me
            game.myself_to_the_right_of_me = elements_to_the_right_abs
            game.myself_to_the_left_of_me = elements_to_the_left_abs
            game.myself_to_the_front_of_me = elements_to_the_top_abs
        
        # if we're going down
        elif(game.direction == Direction.DOWN):
            # if I have something to the right of me
            game.myself_to_the_left_of_me = elements_to_the_right_abs
            game.myself_to_the_right_of_me = elements_to_the_left_abs
            game.myself_to_the_front_of_me = elements_to_the_bottom_abs

        # distances to the walls
        right_wall_abs = (game.w - head.x)/game.w
        left_wall_abs = head.x/game.w
        up_wall_abs = head.y/game.h
        down_wall_abs = (game.h - head.y)/game.h
        
        game.front_wall = 0
        game.right_wall = 0
        game.left_wall = 0
        
        game.food_left = 0
        game.food_right = 0
        game.food_front = 0
        game.food_back = 0
        
        food_left_abs = game.food.x < game.head.x  # food left
        food_right_abs = game.food.x > game.head.x  # food right
        food_up_abs = game.food.y < game.head.y  # food up
        food_down_abs = game.food.y > game.head.y  # food down
        
        # if we're moving right
        if(dir_right):
            game.front_wall = right_wall_abs
            game.right_wall = down_wall_abs
            game.left_wall = up_wall_abs
            
            game.food_left = food_up_abs
            game.food_right = food_down_abs
            game.food_front = food_right_abs
            game.food_back = food_left_abs
                    
        elif(dir_left):
            game.front_wall = left_wall_abs
            game.right_wall = up_wall_abs
            game.left_wall = down_wall_abs

            game.food_left = food_down_abs
            game.food_right = food_up_abs
            game.food_front = food_left_abs
            game.food_back = food_right_abs
        
        elif(dir_up):
            game.front_wall = up_wall_abs
            game.right_wall = right_wall_abs
            game.left_wall = left_wall_abs
            
            game.food_left = food_left_abs
            game.food_right = food_right_abs
            game.food_front = food_up_abs
            game.food_back = food_down_abs
        
        elif(dir_down):
            game.front_wall = down_wall_abs
            game.right_wall = left_wall_abs
            game.left_wall = right_wall_abs
            
            game.food_left = food_right_abs
            game.food_right = food_left_abs
            game.food_front = food_down_abs
            game.food_back = food_up_abs
        
        
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
            
            game.food_left,
            game.food_right,
            game.food_front,
            game.food_back,
            
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
        resultant_state = state_big
    
        return np.array(resultant_state, dtype=int)
    #-------------------------------------------------------------------------------
    def get_2d_state(self, game):
        
        # clearing a state array
        self.state_2d = np.full((33,33), 0.1)
        
        big_2d_field = np.zeros((132,132))
        
        head = game.snake[0]
        food = game.food
        
        # going through all of the snake
        for point in game.snake:
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
        
        head_relative_crop = big_2d_field[head_x - int(self.state_2d.shape[0]) : head_x + int(self.state_2d.shape[0]), head_y - int(self.state_2d.shape[1]) : head_y + int(self.state_2d.shape[1])]
        
        cv2.imshow('Head crop', head_relative_crop)
        
        return head_relative_crop
#-------------------------------------------------------------------------------
    def remember_in_replay_memory(self, state, action, reward, next_state, done):
        """Remember in replay memory"""
        
        # just appending everything in a messy way
        # the ammount of states, actions, rewards - is determined by how long we played the game
        self.replay_memory.append([state, action, reward, next_state, done]) # popleft if MAX_MEMORY is reached
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
        self.epsilon = 5000 - self.game_counter
        
        returned_move = [0,0,0]
        
        # if we start from a newly trained model and we should use EXPLORATION
        if random.randint(0, 9000) < self.epsilon: # train_mode != Train_mode.second_level_network_reward_shaping and
            # getting the number of move where to set 1
            move = random.randint(0, 2)
            returned_move[move] = 1
        
        # use EXPLOITATION
        else:
            state = torch.tensor(state, dtype=torch.float)
            
            if train_mode != Train_mode.primodial_network_training:
                state = state.to('cuda')
                state = torch.unsqueeze(state, 0)
                state = torch.unsqueeze(state, 0)
            
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
    
    get_state_delegate = agent.get_state
    
    if train_mode == Train_mode.primodial_network_training:
        get_state_delegate = agent.get_state
    else:
        get_state_delegate = agent.get_2d_state

    # endless loop
    while True:
        
        # getting a current state from a game
        current_state = get_state_delegate(game)

        # get an action from a model
        next_move = agent.get_action(current_state) # , cortizol, max_prediction, avrg_prediction
        
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
        state_new = get_state_delegate(game)
        

        if train_mode == Train_mode.second_level_network_reward_shaping:
            # getting a small state
            small_state = agent.get_state(game)
            
            small_state = torch.tensor(small_state, dtype=torch.float)
            
            small_model_q_value = torch.max(primodial_model(small_state)).item()
            
            # modifying the reward
            reward = small_model_q_value

        # train short memory
        agent.train_short_memory(current_state, next_move, reward, state_new, done)

        # remember - here we don't know yet the future reward
        agent.remember_in_replay_memory(current_state, next_move, reward, state_new, done)
        
        agent.steps_without_real_future_reward +=1
        
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
                if train_mode == Train_mode.primodial_network_training and rewrite_small_model:
                    agent.model.save('model_11_values.pth')
            
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
            
            if agent.game_counter == 7500:
                print("Trained enough...")

# if we start this file - start the training process
if __name__ == '__main__':
    train()