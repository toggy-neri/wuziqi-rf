import torch
from torch import nn
from torch.nn import functional as F

class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        #residual connection
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)
    
class Network(nn.Module):
    def __init__(self,board_size = 15,num_res_blocks = 10):
        super(Network, self).__init__()

        self.board_size = board_size   
        #init conv block
        self.conv_input = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)

        #residual blocks tower
        self.residual_blocks = nn.Sequential(*[Residual_block(64, 64) for _ in range(num_res_blocks)])

        #Policy head
        self.policy_head = nn.Conv2d(64, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2*board_size*board_size, board_size*board_size)

        #value head
        self.value_head = nn.Conv2d(64, 1, kernel_size=1)
        self.bn_value   = nn.BatchNorm2d(1)          # BN放Conv后
        self.value_fc1  = nn.Linear(board_size*board_size, 256)  # 输入只有225
        self.value_fc2  = nn.Linear(256, 1)

        #后期 加tanh
    def forward(self, x):
        #sharing conv block
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.residual_blocks(x)

        #policy output
        p = F.relu(self.bn_policy(self.policy_head(x)))
        p = p.view(-1, 2*self.board_size*self.board_size)
        p = self.policy_fc(p)

        #value output
        v = F.relu(self.bn_value(self.value_head(x))) 
        v = v.view(-1, self.board_size**2)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        # Value output - 解决 Relu 死亡问题
        # v = self.bn_value(self.value_head(x))
        # v = v.view(x.size(0), -1)
        # v = F.leaky_relu(self.value_fc1(v)) # 确保这里没有全部死掉，或者改用 LeakyReLU
        # v = self.value_fc2(v)   

        return p, v

