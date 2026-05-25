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
    def __init__(self,board_size = 15,num_res_blocks = 10,in_channels = 8,channels = 128):
        super(Network, self).__init__()

        self.board_size = board_size   
        #init conv block
        self.conv_input = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)

        #residual blocks tower
        self.residual_blocks = nn.Sequential(*[Residual_block(channels, channels) for _ in range(num_res_blocks)])

        #Policy head
        self.policy_head = nn.Conv2d(channels, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2*board_size*board_size, board_size*board_size)

        #value head
        self.value_head = nn.Conv2d(channels, 1, kernel_size=1)
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
    
def upgrade_input_channels(model_path, new_model_path, old_channels=4, new_channels=8):
    old_network = Network(board_size=15, channels=old_channels)
    old_network.load_state_dict(torch.load(model_path))
    
    new_network = Network(board_size=15, channels=new_channels)
    
    # 复制除第一层以外的所有权重
    old_state = old_network.state_dict()
    new_state = new_network.state_dict()
    
    for key in old_state:
        if key == 'conv_input.weight':
            # 旧权重 shape: (64, 4, 3, 3)
            # 新权重 shape: (64, 8, 3, 3)
            old_w = old_state[key]  # (64, 4, 3, 3)
            new_w = new_state[key]  # (64, 8, 3, 3)
            # 前4个通道用旧权重，后4个通道用随机初始化（缩小scale避免噪声过大）
            new_w[:, :old_channels, :, :] = old_w
            new_w[:, old_channels:, :, :] *= 0.1  # 缩小新通道初始权重
            new_state[key] = new_w
        else:
            new_state[key] = old_state[key]
    
    new_network.load_state_dict(new_state)
    torch.save(new_network.state_dict(), new_model_path)
    print(f"Saved upgraded model to {new_model_path}")
    return new_network
def upgrade_to_128channels(model_path, new_model_path):
    old_network = Network(board_size=15, num_res_blocks=10, in_channels=8, channels=64)
    old_network.load_state_dict(torch.load(model_path))
    
    new_network = Network(board_size=15, num_res_blocks=10, in_channels=8, channels=128)
    
    old_state = old_network.state_dict()
    new_state = new_network.state_dict()
    
    for key in new_state:
        if key not in old_state:
            # 新增的权重，保持随机初始化并缩小scale
            new_state[key] *= 0.1
            continue
            
        old_w = old_state[key]
        new_w = new_state[key]
        
        if old_w.shape == new_w.shape:
            # 形状一样直接复制（BN层、bias等）
            new_state[key] = old_w
        else:
            # 形状不同的卷积权重，旧的部分复制，新增的部分缩小
            # 例如 (128,64,3,3) <- (64,64,3,3)
            slices = tuple(slice(0, s) for s in old_w.shape)
            new_state[key][slices] = old_w
            # 新增部分已经是随机初始化，乘以0.1降低噪声
            # 把旧部分之外的区域缩小
            mask = torch.zeros_like(new_state[key], dtype=torch.bool)
            mask[slices] = True
            new_state[key][~mask] *= 0.1
    
    new_network.load_state_dict(new_state)
    torch.save(new_network.state_dict(), new_model_path)
    print(f"Saved 128-channel model to {new_model_path}")
    return new_network

if __name__ == '__main__':
    #upgrade_input_channels('runs/alphaTao-v0.5/alphaTao-v0.5.pt', 'runs/alphaTao-v0.5/alphaTao-v0.6.pt',4,8)
    upgrade_to_128channels('runs/alphaTao-v0.6/alphaTao-v0.6.pt', 'runs/alphaTao-v0.6/alphaTao-v0.7.pt')
