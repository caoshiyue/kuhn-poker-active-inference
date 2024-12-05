##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2024-11-21 22:29:27
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

def load_dataset(filename):
    data = torch.load(filename)
    return [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in data]

class MLEModel(nn.Module):
    def __init__(self, theta_dim):
        super(MLEModel, self).__init__()
        # 定义未归一化的参数
        self.theta_unbounded = nn.Parameter(torch.randn(theta_dim))
    
    def forward(self):
        # 使用 Softmax 将参数转换为概率分布
        theta = torch.softmax(self.theta_unbounded, dim=0)
        return theta

if __name__ == "__main__":
    train_data = load_dataset("datasets/obs_action_dataset.pt")
    dataloader = DataLoader(train_data, batch_size=32, shuffle=False)
    encoder = KuhnPokerEncoder()

    num_samples = len(dataloader)
    theta_dim = 54

    # 初始化模型
    agent = ActiveInferenceAgent(encoder)
    theta_model = MLEModel(theta_dim)

    ground_true_prefer=agent.preferred_observations
    # 定义优化器，优化所有需要优化的参数
    optimizer = torch.optim.Adam(theta_model , lr=1e-3)
    criterion = torch.nn.NLLLoss()
    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_obs, batch_actions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_size = batch_obs.size(0)
            
            
            # 前向传播
            # with autocast():
            action_probs_list=[]
            for k in range(batch_size):
                obs=encoder.decode_observation(batch_obs[k].numpy()) 
                action, action_probs = agent.infer_basedon_prefer(obs, theta)
                action_probs=action_probs.unsqueeze(0)
                action_probs_list.append(action_probs)

            # 计算ELBO损失
            action_probs=torch.cat(action_probs_list, dim=0)
            loss = criterion(action_probs, torch.argmax(batch_actions,dim=1))
            # 反向传播和优化
            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_size
        
        avg_loss = epoch_loss / num_samples
        print(f"Epoch {epoch+1}, Average ELBO Loss: {avg_loss:.4f}")
        print()