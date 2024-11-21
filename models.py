import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from kuhn_encode import KuhnPokerEncoder

class TransitionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransitionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)
    
    def infer(self, x):
        p=self.fc(x)
        return binarize_output(p)


class ObservationToStateModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=None):
        super(ObservationToStateModel, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # 使用 Sigmoid 激活函数
        self.temperature = 0.1
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.sigmoid(x)  # 每个类别独立输出概率
        return output
    
    def infer(self, x):
        p=self.forward(x)/self.temperature
        p=torch.softmax(p, dim=0)
        return p

    
def binarize_output(encoded_state, threshold=0.5):
    """
    将连续的编码输出转换为独热编码向量
    """
    return (encoded_state >= threshold).float()


def entropy(prob_dist):
    """
    计算给定概率分布的熵。
    
    Args:
        prob_dist (torch.Tensor): 概率分布，形状为 [..., num_classes]
    
    Returns:
        torch.Tensor: 熵的值，形状为 [...]
    """
    return -torch.sum(prob_dist * torch.log(prob_dist + 1e-10), dim=-1)

def kl_divergence(p, q):
    """
    计算两个概率分布之间的KL散度 KL(p || q)。
    
    Args:
        p (torch.Tensor): 源分布，形状为 [num_classes]
        q (torch.Tensor): 目标分布，形状为 [num_classes]
    
    Returns:
        torch.Tensor: KL散度的值
    """
    return torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10)))