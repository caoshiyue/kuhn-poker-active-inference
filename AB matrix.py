##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2024-11-21 16:59:03
## 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from kuhn_encode import KuhnPokerEncoder
from models import*
import numpy as np



if __name__ == "__main__":
    encoder=KuhnPokerEncoder()
    torch.set_printoptions(
        edgeitems=10,  # 打印的行数和列数的边界
        linewidth=100,  # 每行的最大字符数
        precision=3,    # 小数点后保留的位数
        threshold=1000, # 超过此大小时不截断
        sci_mode=False  # 关闭科学计数法
    )
    threshold = 0.001
    transition_model = torch.load("state_transition_model")
    observation_model = torch.load("obs_model")

    transition_model.eval()
    observation_model.eval()


    P_sa = torch.zeros(encoder.num_states, encoder.num_actions, encoder.num_states)
    with torch.no_grad():
        for s_idx in  range(encoder.num_states):
            state_encoding=encoder.encode_state(encoder.index_to_state[s_idx])
            for a_idx in range(encoder.num_actions):
                action_encoding = encoder.encode_action(encoder.index_to_action[a_idx])
                # 获取 P(s' | s, a) 从 TransitionModel
                combined_input = np.concatenate([state_encoding, action_encoding]) 
                combined_input = torch.tensor(combined_input,dtype=torch.float)
                next_state_prob = transition_model.infer(combined_input)  # [1, num_states]
                P_sa[s_idx, a_idx, :] = next_state_prob.squeeze(0)
    P_sa[P_sa < threshold] = 0
    print(P_sa)
    torch.save(P_sa,"P_sa.pth")


    P_s = torch.ones(encoder.num_states) / encoder.num_states  # [num_states]
    P_o = torch.ones(encoder.num_observations) / encoder.num_observations  # [num_observations]

    # 获取 P(s | o)
    P_s_given_o = torch.zeros(encoder.num_observations, encoder.num_states)  # [num_observations, num_states]

    with torch.no_grad():
        for o_idx in range(encoder.num_observations):
            obs_encoding = encoder.encode_observation(encoder.index_to_observation[o_idx])  # [1, num_observations]
            obs_encoding=torch.tensor(obs_encoding,dtype=torch.float)
            s_prob = observation_model.infer(obs_encoding)  # [1, num_states]
            P_s_given_o[o_idx, :] = s_prob.squeeze(0)
    #处理噪声值
    P_s_given_o[P_s_given_o < threshold] = 0
    # 重新归一化每一行
    row_sums = P_s_given_o.sum(dim=1, keepdim=True)  # 计算每一行的和

    # 防止除以零的情况
    normalized_tensor = P_s_given_o / row_sums if row_sums.max() > 0 else P_s_given_o.clone()


    print("\n观测给定状态的概率矩阵 P(s | o):")
    print(P_s_given_o)

    # 计算 P(o | s) 使用贝叶斯定理
    # P(o | s) = P(s | o) P(o) / P(s)
    # 其中 P(s) = sum_o P(s | o) P(o)

    # 计算 P(s)
    P_s_calculated = torch.matmul(P_s_given_o.t(), P_o)  # [num_states]
    print(f"{P_s_calculated}")
    # 计算 P(o | s)
    # P(o | s) = (P(s | o) * P(o)) / P(s)
    # 需要广播 P(s) 以匹配 P(s | o) * P(o)
    P_o_reshaped = P_o.unsqueeze(0)  # [1, num_observations]
    numerator = P_s_given_o * P_o_reshaped  # [num_observations, num_states]
    # 转置 numerator 和 P_s_calculated 以便相除
    numerator = numerator.t()  # [num_states, num_observations]
    P_o_given_s = numerator / P_s_calculated.unsqueeze(1)  # [num_states, num_observations]

    print("\n观测给定状态的概率矩阵 P(o | s):")
    print(f"{P_o_given_s}")
    torch.save(P_o_given_s,"P_o_given_s.pth")
