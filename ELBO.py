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

class VariationalDistribution(nn.Module):
    def __init__(self, theta_dim):
        super(VariationalDistribution, self).__init__()
        self.theta_dim = theta_dim
        # 参数化均值和对数方差
        self.fc_mu = nn.Linear(theta_dim, theta_dim)
        self.fc_logvar = nn.Linear(theta_dim, theta_dim)
    
    def forward(self, x):
        """
        输入:
            x: Tensor of shape (batch_size, ...)
        输出:
            mu: Tensor of形状 (batch_size, theta_dim)
            logvar: Tensor of形状 (batch_size, theta_dim)
        """
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def sample(self, mu, logvar):
        """
        使用重参数化技巧采样theta
        """
        std = torch.exp(0.5 * logvar)+ 1e-6
        eps = torch.randn_like(std)
        theta_unbounded = mu + eps * std
        theta = F.softmax(theta_unbounded, dim=1)  # 应用 Softmax 确保归一化
        return theta

def load_dataset(filename):
    data = torch.load(filename)
    return [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in data]

def elbo_loss(action_probs, actions, mu, logvar):
    """
    计算ELBO损失
    输入:
        action_probs: Tensor of shape (batch_size, action_dim)
        actions: Tensor of形状 (batch_size,), 整数标签
        mu: Tensor of形状 (batch_size, theta_dim)
        logvar: Tensor of形状 (batch_size, theta_dim)
    输出:
        elbo: 标量
    """
    # 重构损失，使用交叉熵
    recon_loss = F.nll_loss(torch.log(action_probs + 1e-10), actions, reduction='mean')
    
    # KL散度，假设先验p(theta)为标准正态分布
    logvar = torch.clamp(logvar, min=-10, max=10)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / actions.size(0)  # 平均到每个样本
    
    return recon_loss + kl_loss

if __name__ == "__main__":
    train_data = load_dataset("datasets/obs_action_dataset.pt")
    dataloader = DataLoader(train_data, batch_size=32, shuffle=False)
    encoder = KuhnPokerEncoder()

    num_samples = len(dataloader)
    theta_dim = 54

    # 初始化模型
    agent = ActiveInferenceAgent(encoder)
    variational = VariationalDistribution(theta_dim)
    ground_true_prefer=agent.preferred_observations
    # 定义优化器，优化所有需要优化的参数
    optimizer = torch.optim.Adam(variational.parameters() , lr=1e-3)

    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_obs, batch_actions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_size = batch_obs.size(0)
            
            
            # 前向传播
            # with autocast():
            mu, logvar = variational(batch_obs)
            theta = variational.sample(mu, logvar)
            action_probs_list=[]
            for k in range(batch_size):
                obs=encoder.decode_observation(batch_obs[k].numpy()) 
                action, action_probs = agent.infer_basedon_prefer(obs, theta[k])
                action_probs=action_probs.unsqueeze(0)
                action_probs_list.append(action_probs)

            # 计算ELBO损失
            action_probs=torch.cat(action_probs_list, dim=0)
            loss = elbo_loss(action_probs, torch.argmax(batch_actions,dim=1), mu, logvar)
                
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