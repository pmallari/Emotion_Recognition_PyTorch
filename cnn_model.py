import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layer (sees 48x48x1 image tensor)
        self.conv1 = nn.Conv2d(1, 20, 3, padding = 1)
        # Convolutional layer (sees 24x24x20 image tensor)
        self.conv2 = nn.Conv2d(20, 20, 3, padding = 1)
        # Convolutional layer (sees 12x12x20 image tensor)
        self.conv3 = nn.Conv2d(20, 20, 3, padding = 1)
        
        self.maxpool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(20*6*6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)
        
        self.dropout = nn.Dropout(p = 0.25)
        
        self.logsoftmax = nn.LogSoftmax(dim = 1)
        
    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        x = x.view(-1, 6*6*20)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x