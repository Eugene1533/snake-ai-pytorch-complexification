import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    """Liner Q-model"""
    
    def __init__(self, input_size, hidden_size, output_size):
        """Initializer of a class"""
        super().__init__()
        
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
#-------------------------------------------------------------------------------
    def forward(self, x):
        """Forward method of the network"""
        
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x
#-------------------------------------------------------------------------------
    def save(self, file_name='small_pretrained_model.pth'):
        """Save the network"""
        
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
#-------------------------------------------------------------------------------
    def load(self, file_name='small_pretrained_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            raise "model does not exist"

        # getting the full path of the file and loading it
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))
        self.eval()
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
    def train_step(self, state, action, reward, next_state, done):
        """Step of a training process"""
        
        # converting everything to torch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # if we have only one dimention of our state - adding additional one in order to use a model
        if len(state.shape) == 1:
            
            # it must be like (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        # ! We predict quality for each action in this state
        pred = self.model(state)

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
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # setting the Q-value of each action that is predicted by Belman equation
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        
        # pred here is the set of values for actions - how good  they are for a particular state
        # target - what is predicted by Belman equation
        # our predictions and the result of Belman equation must converge
        loss = self.loss_type(target, pred)
        
        # calculating gradients
        loss.backward()
        
        # step of an optimizer
        self.optimizer.step()