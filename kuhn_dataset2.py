##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2024-11-20 11:09:30
## 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import numpy as np
from kuhn import KuhnPokerEnv
from data_gen import generate_state_transition_data
import os
from kuhn_encode import KuhnPokerEncoder
class KuhnPokerDataset(Dataset):
    def __init__(self, state_transition_data, encoder):
        """
        初始化数据集。

        参数：
        - state_transition_data: list of tuples, 每个元组包含 (当前状态, 动作, 下一状态)
        - encoder: 一个KuhnPokerEncoder实例，用于状态和动作的编码
        """
        self.data = state_transition_data
        self.encoder = encoder

    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据点。

        返回：
        - 输入特征 (当前状态one-hot编码, 动作索引)
        - 目标输出 下一状态one-hot编码
        """
        current_state, action, next_state = self.data[idx]
        
        # 编码当前状态和下一状态
        current_state_enc = self.encoder.encode_state(current_state)  # numpy数组
        next_state_enc = self.encoder.encode_state(next_state)        # numpy数组
        
        # 编码动作为整数索引
        action_enc = self.encoder.encode_action(action)               # 整数
        
        combined_input = np.concatenate([current_state_enc, action_enc]) 
        # 将编码后的数据转换为PyTorch张量
        combined_input = torch.tensor(combined_input, dtype=torch.float)
        next_state_enc = torch.tensor(next_state_enc, dtype=torch.float)
        
         # numpy数组，长度为 num_states + num_actions

        # 可以选择将当前状态和动作合并为一个输入特征
        # 例如，拼接当前状态的one-hot向量和动作的one-hot向量
        # 但根据您的需求，此处我们将它们分开返回
        return combined_input, next_state_enc

def save_dataset(dataset, filename):
    """
    保存 PyTorch 数据集到文件。
    参数:
        dataset (Dataset): 需要保存的数据集。
        filename (str): 保存的文件路径。
    """
    data = [(x.numpy(), y.numpy()) for x, y in dataset]
    torch.save(data, filename)
    print(f"数据集已保存到 {filename}")
    
if __name__ == "__main__":

    # 创建 PyTorch 数据集
    env = KuhnPokerEnv()
    encoder = KuhnPokerEncoder()

    state_transition_data = generate_state_transition_data(env, num_games=10000)
    dataset = KuhnPokerDataset(state_transition_data, encoder)

 # 拆分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 保存训练集和测试集
    os.makedirs("datasets", exist_ok=True)
    save_dataset(train_dataset, "datasets/train_dataset.pt")
    save_dataset(test_dataset, "datasets/test_dataset.pt")# 示例使用