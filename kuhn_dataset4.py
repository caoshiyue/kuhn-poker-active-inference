##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2024-11-22 09:02:17
## 
import torch
import torch.nn as nn
from kuhn_encode import KuhnPokerEncoder
from models import*
import numpy as np
import torch.nn.functional as F
from kuhn import KuhnPokerEnv
import random
from active_infer import ActiveInferenceAgent



class Active_infer_Dataset(Dataset):
    def __init__(self, encoder, observations, actions):
        """
        Args:
            observations (list or ndarray): 观测值列表。
            actions (list or ndarray): 行动列表。
        """
        super(Active_infer_Dataset, self).__init__()
        self.encoder = encoder
        # 转换为 PyTorch 张量
        # 假设 observation 和 action 可以转换为数值类型
        # 如果 observation 是字典或其他复杂结构，需根据具体情况调整
        self.observations = observations
        self.actions = actions
        # 如果动作是连续值，可能需要使用 torch.float32

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        """
        返回单个样本，包括 observation 和 action
        """
        return self.observations[idx], self.actions[idx]


def save_dataset(dataset, filename):
    """
    保存 PyTorch 数据集到文件。
    参数:
        dataset (Dataset): 需要保存的数据集。
        filename (str): 保存的文件路径。
    """
    data = [(x, y) for x, y in dataset]
    torch.save(data, filename)
    print(f"数据集已保存到 {filename}")
    

# 示例用法（假设您已经有KuhnPokerEnv和Encoder的实现）
if __name__ == "__main__":
    # 初始化环境和编码器
    observations_list = []
    actions_list = []

    env = KuhnPokerEnv()
    encoder = KuhnPokerEncoder()

    num_states = encoder.num_states  # 从编码器获取
    num_actions = encoder.num_actions
    num_observations = encoder.num_observations

    # 初始化主动推理代理
    agent = ActiveInferenceAgent( encoder=encoder)
    num_episodes = 1000
    max_steps = 10
    player_reward=[0,0] #agent opp
    for episode in range(num_episodes):
        print(f"=== Episode {episode + 1} ===")
        observation = env.reset()
        agent.reset_belief()  # 重置信念
        fisrt=random.randint(0, 1)  #0 则agent先手，1反之 
        # if "Q" in env.get_agent_observation(fisrt)[0] :#! 我们先把Q移除
        #     continue
        agent.update_belief(env.get_agent_observation(fisrt))
        done = False
        step = 0
        while not done and step < max_steps:
            step += 1
            valid_actions = env.get_valid_actions()
            if (fisrt+step)%2==1:
                action,_  = agent.action(observation=observation, valid_actions=valid_actions)
                observations_list.append(encoder.encode_observation(observation))
                actions_list.append(encoder.encode_action(action))
            else:
                action=random.choice(valid_actions)

            # 执行动作，获取环境反馈（假设环境处理对手的动作并返回下一观测）
            next_observation, reward, done, info = env.step(action)
            # 输出当前步的信息
            agent.update_belief(env.get_agent_observation(fisrt))

            observation = next_observation

    dataset = Active_infer_Dataset(encoder,observations_list, actions_list,)
    print(f"Dataset size: {len(dataset)} samples")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    save_dataset(train_dataset,"datasets/obs_action_train_dataset.pt")
    save_dataset(test_dataset,"datasets/obs_action_test_dataset.pt")

    # （可选）创建 DataLoader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 示例：遍历 DataLoader
    for batch_idx, (obs, acts) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Observations: {obs}")
        print(f"  Actions: {acts}")
        # 这里可以加入训练代码
        break  # 仅示例第一个批次