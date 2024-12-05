##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2024-12-04 23:24:34
## 
import torch
import torch.nn as nn
from kuhn_encode import KuhnPokerEncoder
from models import*
import numpy as np
import torch.nn.functional as F
from kuhn import KuhnPokerEnv
import random


class ActiveInferenceAgent:
    def __init__(self,  encoder , initial_belief=None):
        """
        初始化主动推理代理。

        Args:
            env (KuhnPokerEnv): Kuhn扑克环境实例。
            encoder (Encoder): 编码器实例，已实现状态、观测、行动的编码与解码。
            P_sa (torch.Tensor): 状态转移概率矩阵，形状为 [num_states, num_actions, num_states]。
            P_o_given_s (torch.Tensor): 观测概率矩阵，形状为 [num_states, num_observations]。
            initial_belief (torch.Tensor, optional): 初始信念分布，形状为 [num_states]。如果未提供，默认为均匀分布。
        """
        self.encoder = encoder

        self.P_sa = torch.load("P_sa.pth")  # [num_states, num_actions, num_states]
        self.P_o_given_s = torch.load("P_o_given_s.pth")  # [num_states, num_observations]

        self.num_states = encoder.num_states
        self.num_actions = encoder.num_actions
        self.num_observations = encoder.num_observations

        if initial_belief is not None:
            self.q_s = initial_belief / initial_belief.sum()
        else:
            self.q_s = torch.ones(self.num_states) / self.num_states  # [num_states]

        self.history = []
        self.H_A = self.compute_H_A()  # [num_states]
        self.preferred_observations=self.init_preferred_observations()

    def compute_H_A(self):
        """
        计算观测模型 P(o|s) 对应的熵 H(A)。

        Returns:
            torch.Tensor: 每个状态对应的熵，形状为 [num_states]
        """
        return -torch.sum(self.P_o_given_s * torch.log(self.P_o_given_s + 1e-10), dim=1)  # [num_states]
    
    def init_preferred_observations(self):
        preferred_observations = torch.zeros(self.encoder.num_observations)
        for i in range(self.encoder.num_observations):
            player_card=self.encoder.observations[i][0]
            if "J" in player_card:
                preferred_observations[i]=4-self.encoder.observations[i][2]
            if "Q" in player_card:
                preferred_observations[i]=0
            if "K" in player_card:
                preferred_observations[i]=self.encoder.observations[i][2]
            #print(encoder.observations[i][0],encoder.observations[i][1],preferred_observations[i])
        return preferred_observations/sum(preferred_observations)
        

    def select_action(self, valid_actions,obs):
        """
        选择能够最小化未来状态不确定性的行动。

        Args:
            valid_actions (list): 当前有效的行动索引列表。

        Returns:
            int: 选择的行动索引。
        """

        G = torch.zeros(self.num_actions)  # 初始化自由能向量

        for a in range(self.num_actions):
            if a not in valid_actions:
                G[a] = float('inf')  # 无效行动的自由能设为无穷大
                continue

            # 1. 预测下一状态分布 q(s'|a) = q(s) P(s'|s,a)
            q_s_prime = torch.matmul(self.q_s, self.P_sa[:, a, :])  # [num_states]


            # 2. 预测观测分布 P(o|a) = sum_{s'} P(o|s') q(s'|a)
            P_o_a = torch.matmul(self.P_o_given_s.t(), q_s_prime)  # [num_observations]
            P_o_a = P_o_a + 1e-10  # 数值稳定性
            P_o_a = P_o_a / P_o_a.sum()  # 归一化
            
            # print(encoder.index_to_action[a])

            # 3. 计算预测不确定度 Predicted Uncertainty
            # H(A) 是每个状态的熵，[num_states]
            # q(s') 是 [num_states]
            # 预测不确定度为 H(A) dot q(s')
            predicted_uncertainty = torch.dot(self.H_A, q_s_prime)  # 标量

            # 4. 计算预测分歧 Predicted Divergence
            # KL(P(o|a) || P(o))
            predicted_divergence = F.kl_div(P_o_a.log(), self.preferred_observations, reduction='sum')
            
            # 5. 计算自由能 G(a) = Predicted Uncertainty + Predicted Divergence
            G[a] = predicted_uncertainty + predicted_divergence

        # 选择具有最小自由能的行动
        best_action = torch.argmin(G).item()
        # sG=torch.where(torch.isinf(G), torch.tensor(1e3), G)
        temperature=1e5 #! 为了数值好看，正常使用应删掉
        sG = 1 / (G + 1e-10)*temperature
        action_prob = torch.softmax(sG,dim=0)
        # action_prob = sG/sG.sum()

        # top_k_output = torch.topk(action_prob, 3, dim=0).indices.tolist()
        # for  k in top_k_output:
        #     state = encoder.index_to_action.get(k, "未知动作")
        #     print(f"高置信度动作:  {state}，概率 = {action_prob[k]:.6f}")



        # 如果所有有效行动的自由能都是inf，选择第一个有效行动
        if G[best_action] == float('inf'):
            best_action = valid_actions[0]

        return best_action, action_prob

    def update_belief(self,  observation):
        """
        根据观测更新信念。

        Args:
            observation (int): 当前观测。

        Returns:
            torch.Tensor: 更新后的信念分布 [num_states]。
        """
        # 贝叶斯更新：q(s | o) ∝ P(o | s) q(s)

        observation_index= self.encoder.observation_to_index[observation]
        last_action=observation[1]
        prior = self.q_s
        if len(last_action)!=0:
            last_action=last_action[-1]
            action_idx=self.encoder.action_to_index[last_action] 
            P_s_prime = torch.matmul(prior, self.P_sa[:, action_idx, :])  # [num_states]
            prior = P_s_prime

        P_o_given_s = self.P_o_given_s[:, observation_index]  # [num_states]
        posterior = P_o_given_s * prior # [num_states] 
        posterior = posterior / (posterior.sum() + 1e-10)
        self.q_s = posterior

        #top_k_output = torch.topk(posterior, 2, dim=0).indices.tolist()
        # for  k in top_k_output:
        #     state = encoder.index_to_state.get(k, "未知状态")
        #     print(f"高置信度预测: 状态 {state}，概率 = {posterior[k]:.6f}")
        return self.q_s
    
    def reset_belief(self, initial_belief=None):
        """
        重置信念分布。

        Args:
            initial_belief (torch.Tensor, optional): 初始信念分布，形状为 [num_states]。如果未提供，默认为均匀分布。
        """
        if initial_belief is not None:
            self.q_s = initial_belief / initial_belief.sum()
        else:
            self.q_s = torch.ones(self.num_states) / self.num_states  # [num_states]
        self.history = []

    def action(self, observation, valid_actions):
        """
        根据当前观测选择一个行动，并更新信念。

        Args:
            observation (str): 当前观测。
            valid_actions (list): 当前有效的行动索引列表。

        Returns:
            int: 选择的行动索引。
        """
        # 首先，根据观测更新信念
        valid_actions_index=[self.encoder.action_to_index[a] for a in valid_actions]
        
        #self.update_belief(observation_index) #当前观测的索引

        # 选择行动
        action_idx, action_prob = self.select_action(valid_actions_index,observation)
        action = self.encoder.index_to_action[action_idx]
        # 记录历史
        step_record = {
            'action': action,
            'observation': observation,
            'belief': self.q_s.clone().detach()
        }
        self.history.append(step_record)

        return action,action_prob


    def get_valid_actions(self,observation):
        action_history=observation[1]
        if len(action_history)==0:
            return ["check", "bet"]
        last_action = action_history[-1]
        if last_action == "check":
            return ["check", "bet"]
        elif last_action == "bet":
            return ["call", "fold"]

    def infer_basedon_prefer(self,observation,prefer):
        self.preferred_observations=prefer/sum(prefer)
        self.update_belief(observation)
        valid_actions=self.get_valid_actions(observation)
        action,action_prob=self.action(observation,valid_actions)
        return action,action_prob

    def get_history(self):
        """
        获取主动推理的历史记录。

        Returns:
            List of dicts: 每回合的详细历史记录。
        """
        return self.history

# 示例用法（假设您已经有KuhnPokerEnv和Encoder的实现）
if __name__ == "__main__":
    # 假设已导入并实例化KuhnPokerEnv和Encoder
    # from kuhn_poker_env import KuhnPokerEnv
    # from encoder import Encoder

    # 初始化环境和编码器
    env = KuhnPokerEnv()
    encoder = KuhnPokerEncoder()

    # 定义状态转移概率矩阵 P_sa 和观测概率矩阵 P_o_given_s
    # 注意：这些矩阵应根据具体的Kuhn扑克环境和编码器设计
    # 这里提供一个示例，实际应用中应根据训练或环境设计得到

    num_states = encoder.num_states  # 从编码器获取
    num_actions = encoder.num_actions
    num_observations = encoder.num_observations

    # 初始化主动推理代理
    agent = ActiveInferenceAgent( encoder=encoder)
    agent2 = ActiveInferenceAgent( encoder=encoder)
    num_episodes = 10000
    max_steps = 10
    player_reward=[0,0] #agent opp
    for episode in range(num_episodes):
        #print(f"=== Episode {episode + 1} ===")
        observation = env.reset()
        agent.reset_belief()  # 重置信念
        agent2.reset_belief()  # 重置信念
        fisrt=random.randint(0, 1)  #0 则agent先手，1反之 
        agent.update_belief(env.get_agent_observation(fisrt))
        agent2.update_belief(env.get_agent_observation(1-fisrt))
        done = False
        step = 0
       
        while not done and step < max_steps:
            step += 1
            print(f"Turn {step}:")
            #print(f"  Observation: {observation}")
            valid_actions = env.get_valid_actions()

            if (fisrt+step)%2==1:
                #print("  Agent act")
                # 代理选择行动
                action,_ = agent.action(observation=observation, valid_actions=valid_actions)
            else:
                #print("  Oppoent act")
                action,_ = agent2.action(observation=observation, valid_actions=valid_actions)

            # 执行动作，获取环境反馈（假设环境处理对手的动作并返回下一观测）
            next_observation, reward, done, info = env.step(action)
            # 输出当前步的信息
            agent.update_belief(env.get_agent_observation(fisrt))
            agent2.update_belief(env.get_agent_observation(1-fisrt))
            # print(f"  Action: {action}")
            # print(f"  Reward: {reward}")
            # print(f"  Done: {done}")
            # print(f"  Info: {info}")
            #print(f"  Belief: {agent.q_s.numpy()}\n")

            # 更新当前观测
            observation = next_observation
        w=env.determine_winner()
        if env.determine_winner()==0: #先手胜利
            player_reward[fisrt]+=env.get_reward()
        else:
            player_reward[1-fisrt]+=env.get_reward()
    print(f"agent reward:{player_reward[0]},opp reward={player_reward[1]}")

        

