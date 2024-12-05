##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2024-11-22 11:11:37
## 
##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2024-11-22 10:37:15
## 
##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2024-11-21 22:12:33
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
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

import torch.distributions as dist

class VariationalBayesEstimatorDirichlet(nn.Module):
    def __init__(self, theta_dim, prior_alpha=1.0):
        super(VariationalBayesEstimatorDirichlet, self).__init__()
        # 定义Dirichlet的alpha参数，确保其为正
        self.alpha_logits = nn.Parameter(torch.ones(theta_dim))  # 初始化为1
        self.prior_alpha = prior_alpha  # 可以根据先验调整

    def forward(self):
        # 使用Softplus确保alpha为正
        alpha = torch.nn.functional.softplus(self.alpha_logits)
        theta_dist = dist.Dirichlet(alpha)
        return theta_dist
    

def load_dataset(filename):
    data = torch.load(filename)
    return [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in data]

def compute_elbo_dirichlet(model, agent, obs_batch, action_batch, prior_alpha=0.5):
    """
    计算使用Dirichlet变分分布的ELBO
    """
    theta_dist = model()  # Dirichlet分布
    # 使用重参数化采样
    theta = theta_dist.rsample()

    # 计算 log p(data | theta)
    log_p_data = 0.0
    for obs, action in zip(obs_batch, action_batch):
        action_probs = agent.infer_basedon_prefer(obs, theta)
        log_p_data += torch.log(action_probs[action] + 1e-10)  # 避免 log(0)

    # 计算先验 log p(theta)，假设先验也是Dirichlet
    prior = dist.Dirichlet(torch.ones_like(theta) * prior_alpha)
    log_p_theta = prior.log_prob(theta)

    # 计算变分分布 log q(theta)
    log_q_theta = theta_dist.log_prob(theta)

    # ELBO = E_q[log p(data|theta)] + E_q[log p(theta)] - E_q[log q(theta)]
    # 由于我们已经采样theta，通过单个样本近似
    elbo = log_p_data + log_p_theta - log_q_theta
    return -elbo  # 由于我们要最小化负ELBO


if __name__ == "__main__":
    train_data = load_dataset("datasets/obs_action_dataset.pt")
    dataloader = DataLoader(train_data, batch_size=32, shuffle=False)
    encoder = KuhnPokerEncoder()

    num_samples = len(dataloader)
    theta_dim = 54

    # 初始化模型
    agent = ActiveInferenceAgent(encoder)
    variational = VariationalBayesEstimatorDirichlet(theta_dim)
    ground_true_prefer=agent.preferred_observations
    # 定义优化器，优化所有需要优化的参数
    optimizer = torch.optim.Adam(variational.parameters() , lr=0.001)

    prior_alpha=0.5
    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0.0


        for batch_obs, batch_actions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_size = batch_obs.size(0)
            log_p_data=0.0
            theta_dist = variational()  # Dirichlet分布
            # 使用重参数化采样
            theta = theta_dist.rsample()
            action_probs_list=[]
            for k in range(batch_size):
                obs=encoder.decode_observation(batch_obs[k].numpy()) 
                agent.reset_belief()
                action, action_probs = agent.infer_basedon_prefer(obs, ground_true_prefer)
                action_probs=action_probs.unsqueeze(0)
                action_probs_list.append(action_probs)
                #print(action_probs, batch_actions[k])

            # 计算ELBO损失
            action_probs=torch.cat(action_probs_list, dim=0)
            ba=torch.argmax(batch_actions,dim=1)
            log_p_data += torch.mean(torch.log(action_probs[ba] + 1e-10))

            # 计算先验 log p(theta)，假设先验也是Dirichlet
            prior = dist.Dirichlet(torch.ones_like(theta) * prior_alpha)
            log_p_theta = prior.log_prob(theta)

            # 计算变分分布 log q(theta)
            log_q_theta = theta_dist.log_prob(theta)

            # ELBO = E_q[log p(data|theta)] + E_q[log p(theta)] - E_q[log q(theta)]
            # 由于我们已经采样theta，通过单个样本近似
            elbo = log_p_data + log_p_theta - log_q_theta
            loss = -elbo
            # 反向传播和优化
            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss.backward()
            torch.nn.utils.clip_grad_norm_(variational.parameters(), max_norm=1.0)
            optimizer.step()
        
            epoch_loss += loss.item() * batch_size
        
        avg_loss = epoch_loss / num_samples
        print(f"Epoch {epoch+1}, Average ELBO Loss: {avg_loss:.4f}")
        print()