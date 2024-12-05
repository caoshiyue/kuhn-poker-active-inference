##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2024-11-22 09:27:45
## 
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

if __name__ == "__main__":
    torch.set_printoptions(
    edgeitems=10,  # 打印的行数和列数的边界
    linewidth=100,  # 每行的最大字符数
    precision=5,    # 小数点后保留的位数
    threshold=1000, # 超过此大小时不截断
    sci_mode=False  # 关闭科学计数法
    )
    train_data = load_dataset("datasets/obs_action_dataset.pt")
    dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    encoder = KuhnPokerEncoder()

    num_samples = len(dataloader)
    theta_dim = 54

    # 初始化模型
    agent = ActiveInferenceAgent(encoder)
    theta = torch.randn(theta_dim, requires_grad=True)  # 独立的theta参数

    ground_true_prefer=agent.preferred_observations
    print(ground_true_prefer)
    print(torch.softmax(theta,dim=0))
    # 定义优化器，优化所有需要优化的参数
    optimizer = torch.optim.Adam([theta] , lr=0.01)
    criterion = torch.nn.NLLLoss()
    regularization_strength = 1e-4
    # 训练循环
    num_epochs = 80
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_obs, batch_actions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_size = batch_obs.size(0)
            
            # 前向传播
            # with autocast():
            action_probs_list=[]
            for k in range(batch_size):
                obs=encoder.decode_observation(batch_obs[k].numpy()) 
                agent.reset_belief()
                action, action_probs = agent.infer_basedon_prefer(obs, torch.softmax(theta,dim=0))
                #print(action_probs,batch_actions[k])
                action_probs=action_probs.unsqueeze(0)
                action_probs_list.append(action_probs)

            # 计算ELBO损失
            action_probs=torch.cat(action_probs_list, dim=0)

            batch_actions = torch.argmax(batch_actions,dim=1).view(-1, 1)  # shape [batch_size, 1]
            taken_action_probs = torch.gather(action_probs, dim=1, index=batch_actions).squeeze(1)  # shape [batch_size]
            # 计算负对数似然
            loss = -torch.log(taken_action_probs + 1e-10).mean()+regularization_strength * torch.norm(theta)  # 避免 log(0)

            #loss = criterion(action_probs, torch.argmax(batch_actions,dim=1))
            # 反向传播和优化
            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_size
        
        avg_loss = epoch_loss / num_samples
        print(f"Epoch {epoch+1}, Average ELBO Loss: {avg_loss:.4f}")
        print(ground_true_prefer)
        print(torch.softmax(theta,dim=0))