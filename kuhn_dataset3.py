import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from kuhn_encode import KuhnPokerEncoder
from data_gen import generate_state_transition_data
import os
from kuhn import KuhnPokerEnv
import random

# 定义 ObservationDataset 类
class ObservationDataset(Dataset):
    def __init__(self, state_transition_data, encoder):
        """
        初始化数据集。

        参数：
        - state_transition_data: list of tuples, 每个元组包含 (当前状态, 动作, 下一状态)
        - encoder: 一个 KuhnPokerEncoder 实例，用于状态和观测的编码
        """
        self.data = state_transition_data
        self.encoder = encoder
        self.num_players = 2  

    def __len__(self):
        return len(self.data) * self.num_players

    def __getitem__(self, idx):
        """
        获取指定索引的数据点。

        返回：
        - 输入特征 (玩家观测的one-hot编码)
        - 目标输出 状态类别的one-hot编码
        """
        # 计算对应的数据点和玩家
        state_idx = idx // self.num_players
        player_position = (idx % self.num_players) + 1  # 1 或 2

        # 提取目标状态（第三个元素）
        current_state, _, next_state  = self.data[state_idx]
        if random.random()<0.5: #! 注意，这不可取，因为数据随机
            current_state=next_state
        card=current_state[0]
        ah=current_state[1]
        b=current_state[2]
        card1=(card[0],"MASK")
        card2=("MASK",card[1])
        obs1=(card1,ah,b)
        obs2=(card2,ah,b)
        # 编码目标状态
        state_enc = self.encoder.encode_state(current_state)  # numpy数组
        

        # 构造并编码玩家的观测
        if player_position==1:
            observation_enc = self.encoder.encode_observation(obs1)  # numpy数组
        else:
            observation_enc = self.encoder.encode_observation(obs2)  # numpy数组

        # 将编码后的数据转换为PyTorch张量
        observation_enc = torch.tensor(observation_enc, dtype=torch.float)
        state_enc = torch.tensor(state_enc, dtype=torch.float)

        return observation_enc, state_enc


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
    # 定义 ObservationDataset 和 DataLoader
    state_transition_data = generate_state_transition_data(env, num_games=10000)

    observation_dataset = ObservationDataset(state_transition_data, encoder)

    train_size = int(0.8 * len(observation_dataset))
    test_size = len(observation_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(observation_dataset, [train_size, test_size])

    os.makedirs("datasets", exist_ok=True)
    save_dataset(train_dataset, "datasets/obs_train_dataset.pt")
    save_dataset(test_dataset, "datasets/obs_test_dataset.pt")# 示例使用


    batch_size = 4  # 示例批大小，请根据需要调整
    observation_dataloader = DataLoader(observation_dataset, batch_size=batch_size, shuffle=True)

    # 迭代 DataLoader，查看数据
    for batch_idx, (inputs, targets) in enumerate(observation_dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"输入特征形状: {inputs.shape}")    # [batch_size, num_observations]
        print(f"状态类别编码形状: {targets.shape}")  # [batch_size, num_states]
        print("-" * 50)
