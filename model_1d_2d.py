import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from game import train_stage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Composite_Net(nn.Module):
    """Composite 1d and 2d model Q-model"""
    
    def __init__(self, linear_2d_size):
        """Initializer of a cortex part"""
        super().__init__()
        
        output_size_2d = 16
        hidden_size_2d = 256        
        input_size_1d = 11 # 19
        hidden_size_1d = 256
        output_size_1d = 3
        input_size_aggregator = output_size_2d + output_size_1d
        output_size_aggregator = 3
        
        self.is_2d_mode = True
        
        self._train_stage = train_stage.Zeros_1
        """Internal trainging stage"""
        
        # 2d part
        self.conv1_2d = nn.Conv2d(1,16, kernel_size = 3, padding = 1)
        self.conv2_2d = nn.Conv2d(16, 8, kernel_size = 3, padding = 1)
        self.linear_1_2d = nn.Linear(linear_2d_size, hidden_size_2d)
        self.linear_2_2d = nn.Linear(hidden_size_2d, output_size_2d)
        
        # 1d part
        self.linear_1 = nn.Linear(input_size_1d, hidden_size_1d)
        self.linear_2 = nn.Linear(hidden_size_1d, output_size_1d)
        
        # aggregator
        self.aggregator = nn.Linear(input_size_aggregator, output_size_aggregator)
        
        # if we're in 2d mode
        if self.is_2d_mode:    
            # freezing 1d path
            for param in self.linear_2.parameters():
                param.requires_grad = False
                
            for param in self.linear_1.parameters():
                param.requires_grad = False
        else:
            for param in self.conv1_2d.parameters():
                param.requires_grad = False
                
            for param in self.conv2_2d.parameters():
                param.requires_grad = False
                
            for param in self.linear_1_2d.parameters():
                param.requires_grad = False
                
            for param in self.linear_2_2d.parameters():
                param.requires_grad = False
        
#-------------------------------------------------------------------------------
    def forward(self, input_1d, input_2d):
        """Forward method of the network"""
        
        # 2d pass
        x_2d = F.relu(self.conv1_2d(input_2d))        
        x_2d = F.relu(self.conv2_2d(x_2d))
        x_2d = torch.flatten(x_2d, 1)
        x_2d = self.linear_1_2d(x_2d)
        x_2d = F.relu(x_2d)
        x_2d = self.linear_2_2d(x_2d)
        
        # 1d pass
        x_1d = F.relu(self.linear_1(input_1d))
        x_1d = self.linear_2(x_1d)

        # creating a placeholder with zeros for 2d input
        feature_vector_2d_placeholder = torch.zeros((x_1d.shape[0], 1, x_2d.shape[1]))
        feature_vector_2d_placeholder = feature_vector_2d_placeholder.to(device)
        feature_vector_2d_placeholder.requires_grad = False
        
        # choosing between the real 2d feature vector or a placeholder
        final_2d_vector = None
        
        # # if we work with initial 1d mode
        if self._train_stage == train_stage.Zeros_1:
            final_2d_vector = feature_vector_2d_placeholder
            
        else:
            final_2d_vector = x_2d
            final_2d_vector = torch.unsqueeze(final_2d_vector, 1)
        
        # aggregating two outputs
        aggregated_1d_and_2d = torch.cat((x_1d, final_2d_vector), dim=2)
        aggregated_output = self.aggregator(aggregated_1d_and_2d)

        return aggregated_output
#-------------------------------------------------------------------------------
    def save(self, file_name='model_support_network.pth'):
        """Save the network"""
        
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
#-------------------------------------------------------------------------------
    def load(self, file_name='model_support_network.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            raise "model does not exist"

        # getting the full path of the file and loading it
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))
        self.eval()
#-------------------------------------------------------------------------------
    def set_training_stage(self, training_stage: train_stage):
        """Set the training stage"""
        
        self._train_stage = training_stage
        
        # For the zeros
        if self._train_stage == train_stage.Zeros_1:
            
            # freezing the 2d part
            for param in self.conv1_2d.parameters():
                param.requires_grad = False
                
            for param in self.conv2_2d.parameters():
                param.requires_grad = False
                
            for param in self.linear_1_2d.parameters():
                param.requires_grad = False
                
            for param in self.linear_2_2d.parameters():
                param.requires_grad = False
            
            # unfreezing the 1d part
            for param in self.linear_2.parameters():
                param.requires_grad = True
                
            for param in self.linear_1.parameters():
                param.requires_grad = True
            
        # For the noise
        elif self._train_stage == train_stage.Noise_2:
            
            # freezing the 2d part
            for param in self.conv1_2d.parameters():
                param.requires_grad = False
                
            for param in self.conv2_2d.parameters():
                param.requires_grad = False
                
            for param in self.linear_1_2d.parameters():
                param.requires_grad = False
                
            for param in self.linear_2_2d.parameters():
                param.requires_grad = False
            
            # unfreezing the 1d part
            for param in self.linear_2.parameters():
                param.requires_grad = True
                
            for param in self.linear_1.parameters():
                param.requires_grad = True
        
        # For the involving    
        elif self._train_stage == train_stage.Involving_3:
            
            # unfreezing the 2d part
            for param in self.conv1_2d.parameters():
                param.requires_grad = True
                
            for param in self.conv2_2d.parameters():
                param.requires_grad = True
                
            for param in self.linear_1_2d.parameters():
                param.requires_grad = True
                
            for param in self.linear_2_2d.parameters():
                param.requires_grad = True
            
            # freezing the 1d part
            for param in self.linear_2.parameters():
                param.requires_grad = False
                
            for param in self.linear_1.parameters():
                param.requires_grad = False
        
        # For the fine tunning
        elif self._train_stage == train_stage.Both_heads_4:
            
            # unfreezing the 2d part
            for param in self.conv1_2d.parameters():
                param.requires_grad = True
                
            for param in self.conv2_2d.parameters():
                param.requires_grad = True
                
            for param in self.linear_1_2d.parameters():
                param.requires_grad = True
                
            for param in self.linear_2_2d.parameters():
                param.requires_grad = True
            
            # unfreezing the 1d part
            for param in self.linear_2.parameters():
                param.requires_grad = True
                
            for param in self.linear_1.parameters():
                param.requires_grad = True
                # For the fine tunning
        
        # For frozen convolutional part      
        elif self._train_stage == train_stage.Frozen_conv_5:
            
            # unfreezing the 2d part
            for param in self.conv1_2d.parameters():
                param.requires_grad = False
                
            for param in self.conv2_2d.parameters():
                param.requires_grad = False
                
            for param in self.linear_1_2d.parameters():
                param.requires_grad = True
                
            for param in self.linear_2_2d.parameters():
                param.requires_grad = True
            
            #freezing the 1d part
            for param in self.linear_2.parameters():
                param.requires_grad = False
                
            for param in self.linear_1.parameters():
                param.requires_grad = False
#-------------------------------------------------------------------------------
    def get_training_stage(self):
        """Get the training stage"""
        
        return self._train_stage       
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class QTrainer:
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
    def train_step(self, state_1d, state_2d, action, reward, next_state_1d, next_state_2d, done):
        """Step of a training process"""
        
        if isinstance(state_1d, tuple):          
            state_1d = torch.tensor(state_1d, dtype=torch.float)
            state_1d = torch.unsqueeze(state_1d, dim=1)
            
        
        if isinstance(next_state_1d, tuple):
            #next_state = torch.tensor(next_state, dtype=torch.float)
            next_state_1d = torch.stack(next_state_1d)
        
        # converting everything to torch tensors
        state_1d = torch.tensor(state_1d, dtype=torch.float)
        state_1d = state_1d.to(device)
        
        state_2d = torch.tensor(state_2d, dtype=torch.float)
        state_2d = state_2d.to(device)
        
        next_state_1d = torch.tensor(next_state_1d, dtype=torch.float)
        next_state_1d = next_state_1d.to(device)
        next_state_2d = torch.tensor(next_state_2d, dtype=torch.float)
        next_state_2d = next_state_2d.to(device)
        
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        
        if len(state_1d.shape) == 1:
            
            # it must be like (1, x)
            state_1d = torch.unsqueeze(state_1d, 0)
            state_1d = torch.unsqueeze(state_1d, 0)
            next_state_1d = torch.unsqueeze(next_state_1d, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        if len(state_2d.shape) == 2:
            state_2d = torch.unsqueeze(state_2d, 0)
            state_2d = torch.unsqueeze(state_2d, 0)
        
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
        pred = self.model(state_1d, state_2d)
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
                
                next_state_single_1d = next_state_1d[idx]
                next_state_single_2d = next_state_2d[idx]
                
                
                if len(next_state_single_1d.shape) ==2:
                    next_state_single_1d = torch.unsqueeze(next_state_single_1d, 0)
                    
                if len(next_state_single_2d.shape) ==3:
                    next_state_single_2d = torch.unsqueeze(next_state_single_2d, 0)

                next_state_prediction = self.model(next_state_single_1d, next_state_single_2d)
                
                next_state_reward = torch.max(next_state_prediction)
                Q_new = current_reward + self.gamma * next_state_reward

            # setting the Q-value of each action that is predicted by Belman equation
            index_of_best_action = torch.argmax(action[idx]).item()
            
            if len(target.shape) == 3:
                target = torch.squeeze(target, dim=1)
            
            target[idx][index_of_best_action] = Q_new

        self.optimizer.zero_grad()
        
        # pred here is the set of values for actions - how good are they for a particular state
        # target - what is predicted by Belman equation
        # !!! our predictions and the result of Belman equation must converge !!!
        loss = self.loss_type(target, pred)
        
        # calculating gradients
        loss.backward()
        
        # step of an optimizer
        self.optimizer.step()