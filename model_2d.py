import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from game import train_stage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model2d(nn.Module):
    """Just 2d Q-model"""
    
    def __init__(self, linear_2d_size):
        """Initializer of a cortex part"""
        super().__init__()
        
        hidden_size_2d = 256        
        output_size_aggregator = 3
        
        self.is_2d_mode = True
        
        
        # 2d part
        self.conv1_2d = nn.Conv2d(1,16, kernel_size = 3, padding = 1)
        self.conv2_2d = nn.Conv2d(16, 8, kernel_size = 3, padding = 1)
        self.linear_1_2d = nn.Linear(linear_2d_size, hidden_size_2d)
        self.linear_2_2d = nn.Linear(hidden_size_2d, output_size_aggregator)
        
#-------------------------------------------------------------------------------
    def forward(self, input_2d):
        """Forward method of the network"""
        
        # 2d pass
        x_2d = F.relu(self.conv1_2d(input_2d))        
        x_2d = F.relu(self.conv2_2d(x_2d))
        x_2d = torch.flatten(x_2d, 1)
        x_2d = self.linear_1_2d(x_2d)
        x_2d = F.relu(x_2d)
        x_2d = self.linear_2_2d(x_2d)

        return x_2d
#-------------------------------------------------------------------------------
    def save(self, file_name='model_just_2d.pth'):
        """Save the network"""
        
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
#-------------------------------------------------------------------------------
    def load(self, file_name='model_just_2d.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            raise "model does not exist"

        # getting the full path of the file and loading it
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))
        self.eval()
#-------------------------------------------------------------------------------
    def get_training_stage(self):
        """Get the training stage"""
        
        return self._train_stage       
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class QTrainer_2d:
    """Trainer of a model in the realm of a Belman equation"""
    
    def __init__(self, model, lr, gamma):
        """Initializer of a class"""
        # learning rate
        self.lr = lr
        
        # gamma - from the Belman equation
        self.gamma = gamma
        
        # model
        self.model = model
        
        # optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        # criterion for loss
        self.loss_type = nn.MSELoss()
#-------------------------------------------------------------------------------    
    def set_lr(self, lr):
        """Set lr"""
        
        self.lr = lr
        
        assert(self.model != None, "There must be a model!")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
#-------------------------------------------------------------------------------
    def train_step(self, state_2d, action, reward, next_state_2d, done):
        """Step of a training process"""
        
        state_2d = torch.tensor(state_2d, dtype=torch.float)
        state_2d = state_2d.to(device)
        
        next_state_2d = torch.tensor(next_state_2d, dtype=torch.float)
        next_state_2d = next_state_2d.to(device)
        
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        
        if len(state_2d.shape) == 2:
            state_2d = torch.unsqueeze(state_2d, 0)
            state_2d = torch.unsqueeze(state_2d, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        
        if len(next_state_2d.shape) == 2:
            next_state_2d = torch.unsqueeze(next_state_2d, 0)
            next_state_2d = torch.unsqueeze(next_state_2d, 0)
            
        # !!! maybe just one condition and casting or viewing everything insted of unsqueezing !!!
        if len(state_2d.shape) == 3:
            state_2d = torch.unsqueeze(state_2d, 1)
        
        if len(next_state_2d.shape) == 3:
            next_state_2d = torch.unsqueeze(next_state_2d, 1)

        # 1: predicted Q values with current state
        # ! We predict quality for each action in this state
        self.model.eval()
        pred = self.model(state_2d)
        self.model.train()
        
        #pred = torch.unsqueeze(pred, 0)
        
        # cloning the predictions
        target = pred.clone()
        
        
        # done is one dimensional ???
        for idx in range(len(done)):
            
            # getting the current reward
            Q_new = reward[idx]
            
            # if we haven't finished yet
            if not done[idx]:
                
                # getting the new Q value
                # current reward plus gamma multiplied by the expected reward of a new state
                # the expected reward from the next state we get predicted from a model
                
                current_reward = reward[idx]
                
                next_state_single_2d = next_state_2d[idx]
                

                if len(next_state_single_2d.shape) ==3:
                    next_state_single_2d = torch.unsqueeze(next_state_single_2d, 0)
                
                next_state_prediction = self.model(next_state_single_2d)
                
                
                next_state_reward = torch.max(next_state_prediction)
                Q_new = current_reward + self.gamma * next_state_reward

            # setting the Q-value of each action that is predicted by Belman equation
            index_of_best_action = torch.argmax(action[idx]).item()
            
            if len(target.shape) == 3:
                target = torch.squeeze(target, dim=1)
            
            target[idx][index_of_best_action] = Q_new
			

            # setting the Q-value of each action that is predicted by Belman equation
            #target[idx][torch.argmax(action[idx]).item()] = Q_new.item()
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        
        # pred here is the set of values for actions - how good are they for a particular state
        # target - what is predicted by Belman equation
        # !!! our predictions and the result of Belman equation must converge !!!
        loss = self.loss_type(target, pred)
        
        # calculating gradients
        loss.backward()
        
        # step of an optimizer
        self.optimizer.step()