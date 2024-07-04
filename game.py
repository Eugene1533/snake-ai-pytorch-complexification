import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# initializing pygame
pygame.init()

# setting the font
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    """Enum direction"""
    
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
class train_stage(Enum):
    """Enum train mode"""
    
    Zeros_1 = 1
    Noise_2 = 2
    Involving_3 = 3
    Both_heads_4 = 4
    Frozen_conv_5 = 5
    
class conv_loading_mode(Enum):
    """Enum convolution loading mode"""
    
    Dense_small_network = 1
    Conv_network = 2

# named tuple for points
Point = namedtuple('Point', 'x, y')

# colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
ORANGE_RED = (255, 69, 0)
Chartreuse_color = (127, 255, 0)
Gold_color = (255, 215, 0)

BLOCK_SIZE = 20
SPEED = 100000000000000

display_distances_to_walls = False

display_direction_to_myself = False

class SnakeGameAI:
    """Snake game AI class"""

    def __init__(self, w=640, h=640):
        """Initializer"""
        
        self.w = w
        self.h = h
        
        self.myself_to_the_right_of_me = False
        self.myself_to_the_left_of_me = False
        self.myself_to_the_front_of_me = False
        
        self.front_wall = 0
        self.right_wall = 0
        self.left_wall = 0
        
        self.last_turn_right = False
        self.last_turn_left = False
        
        # initialization of a display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake deep RL')
        
        # setting the clock
        self.clock = pygame.time.Clock()
        
        # reseting the game - start from the beginning
        self.reset()
#-------------------------------------------------------------------------------
    def reset(self):
        """Reseting the game"""
        
        # initializing game state
        # moving into the right direction at the beginning
        self.direction = Direction.RIGHT

        # the head will be at he center of a screen
        self.head = Point(self.w/2, self.h/2)
        
        # !!! a snake is a list of points where the first one is a head !!!
        self.snake = [self.head,
                    Point(self.head.x-BLOCK_SIZE, self.head.y),
                    Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        # zeroing score
        # score - how long a snake is
        # it's not reward
        self.score = 0
        
        # we have no food
        self.food = None
        
        # placing food initially
        self._place_food()
        
        # discarding the frame iteration number
        self.frame_iteration_number = 0
        
        # reseting new parameters
        self.myself_to_the_right_of_me = False
        self.myself_to_the_left_of_me = False
        self.myself_to_the_front_of_me = False
        
        self.front_wall = 0
        self.right_wall = 0
        self.left_wall = 0
        
        self.last_turn_right = False
        self.last_turn_left = False
#-------------------------------------------------------------------------------
    def _place_food(self):
        """Placing the food"""
        
        # getting random coordinates
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        
        # !!! food is just a point
        self.food = Point(x, y)
        
        # preventing food being inside a snake
        if self.food in self.snake:
            self._place_food()
#-------------------------------------------------------------------------------
    def play_iteration(self, action):
        """Play a single iteration given an action - returns reward"""
        
        # incrementing the iteration number
        self.frame_iteration_number += 1
        
        # 1. if we press escape - quit the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        # calling an internal method for moving
        self._move(action)
        
        # inserting the head to the first position
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        
        # if we've got a collision or iterate 100 times bigger than the len of snake
        if self.is_collision():
            game_over = True
            
            reward = -10
            
            return reward, game_over, self.score
        
        elif self.frame_iteration_number > 100*len(self.snake):
            game_over = True
            
            reward = -10
            
            return reward, game_over, self.score

        # 4. place new food or just move
        # if the head of a snake is with the food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            # ??? removing last element ???
            # moving in such a way ???
            self.snake.pop()
        
        # 5. update ui and clock
        
        # updating ui
        self._update_ui()
        
        # next tick of clock
        self.clock.tick(SPEED)
        
        # 6. return game over and score
        return reward, game_over, self.score
#-------------------------------------------------------------------------------
    def is_collision(self, pt=None):
        """Collision detection, default poit - head"""
        
        if pt is None:
            pt = self.head
        
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        
        # hits itself - it's the commonplace
        # it's necessary to differenciate between hitting itself and a wall
        if pt in self.snake[1:]:
            return True

        return False
#-------------------------------------------------------------------------------
    def _update_ui(self):
        """Updating ui"""
        
        # return
        
        # filling display black
        self.display.fill(BLACK)

        # for each element in a snake
        for pt in self.snake:
            
            # draw the output rectangle
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            
            # draw the internal rectangle
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # draw the head of a snake
        pygame.draw.rect(self.display, RED, pygame.Rect(self.snake[0].x+4, self.snake[0].y+4, 5, 5))

        # draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # displaying the score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        
        if(display_direction_to_myself):
            # displaying the presence to the right
            if(self.myself_to_the_right_of_me): 
                text = font.render("Right", True, Chartreuse_color)
                self.display.blit(text, [450, 350])
            
            # displaying the presence to the right
            if(self.myself_to_the_left_of_me): 
                text = font.render("Left", True, Chartreuse_color)
                self.display.blit(text, [450, 400])
            
            # displaying the presence in front
            if(self.myself_to_the_front_of_me): 
                text = font.render("Front", True, Chartreuse_color)
                self.display.blit(text, [450, 450])
        
        # if we should display distnances to the walls
        if(display_distances_to_walls):
            # displaying the distance to the front wall
            text = font.render("forward_wall: " + str(self.front_wall), True, Gold_color)
            self.display.blit(text, [10, 350])
            
            # displaying the distance to the right wall
            text = font.render("right_wall: " + str(self.right_wall), True, Gold_color)
            self.display.blit(text, [10, 400])
            
            # displaying the distance to the left wall
            text = font.render("left_wall: " + str(self.left_wall), True, Gold_color)
            self.display.blit(text, [10, 450])
            
            if(self.last_turn_right):
                text = font.render("Last turn right", True, Gold_color)
                self.display.blit(text, [10, 250])
                
            if(self.last_turn_left):
                text = font.render("Last turn left", True, Gold_color)
                self.display.blit(text, [10, 300])
        
        
        # udates the contents of the entire display
        pygame.display.flip()
#-------------------------------------------------------------------------------
    def _move(self, action):
        """Move according to action enum"""
        
        # [straight, right, left]

        clock_wise_list = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        
        # getting an index of the direction in a clockwise list
        idx = clock_wise_list.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise_list[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise_list[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise_list[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        
        # depending on the direction - increment the position of a head
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        # setting the position of a head
        self.head = Point(x, y)