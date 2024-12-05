##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2024-12-04 23:51:36
## 
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

class CategoricalHingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(CategoricalHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, logits, target):
        # logits: [batch_size, num_classes]
        # target: [batch_size]，为类别索引
        correct_class_scores = logits[torch.arange(logits.size(0)), target].unsqueeze(1)
        margins = logits - correct_class_scores + self.margin
        margins[torch.arange(logits.size(0)), target] = 0  # 不考虑正确类别
        loss = torch.clamp(margins, min=0).max(dim=1)[0]
        return loss.mean()

class SoftmaxHingeLoss(nn.Module):
    def __init__(self, margin=0.01):
        """
        初始化多类铰链损失

        参数:
            margin (float): 边界值，用于区分正确类别与其他类别的概率
        """
        super(SoftmaxHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, predictions, targets):
        """
        前向传播

        参数:
            predictions (Tensor): 模型的预测输出，已经过 softmax, 形状 [batch_size, num_classes]
            targets (Tensor): 真实标签，形状 [batch_size]
        
        返回:
            loss (Tensor): 计算得到的损失值
        """
        # 提取正确类别的概率，形状 [batch_size, 1]
        correct_class_prob = predictions[torch.arange(predictions.size(0)), targets].unsqueeze(1)
        
        # 计算每个样本每个类别的 margin 损失
        margin_loss = predictions - correct_class_prob + self.margin
        
        # 将正确类别的损失设置为 0，因为我们不想计算它
        margin_loss[torch.arange(predictions.size(0)), targets] = 0
        
        # 计算每个样本的最大损失
        loss_per_sample, _ = torch.max(margin_loss, dim=1)
        
        # 取负号，因为我们希望正确类别的概率更高
        loss = torch.clamp(loss_per_sample, min=0).mean()
        
        return loss


if __name__ == "__main__":
    torch.set_printoptions(
    edgeitems=10,  # 打印的行数和列数的边界
    linewidth=100,  # 每行的最大字符数
    precision=3,    # 小数点后保留的位数
    threshold=1000, # 超过此大小时不截断
    sci_mode=False  # 关闭科学计数法
    )
    train_data = load_dataset("datasets/obs_action_train_dataset.pt")
    test_data = load_dataset("datasets/obs_action_test_dataset.pt")

    dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)
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
    criterion = SoftmaxHingeLoss()
    regularization_strength = 1e-4

    # 训练循环
    num_epochs = 4
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
                action, action_probs = agent.infer_basedon_prefer(obs, torch.softmax(theta,dim=0)) #ground_true_prefer)#
                #print(action_probs,batch_actions[k])
                action_probs=action_probs.unsqueeze(0)
                action_probs_list.append(action_probs)
            #print(action_probs, batch_actions[-1])
            # 计算ELBO损失
            action_probs=torch.cat(action_probs_list, dim=0)

            loss= criterion(action_probs,torch.argmax(batch_actions,dim=1))+regularization_strength * torch.norm(theta) 
            #loss = criterion(action_probs, torch.argmax(batch_actions,dim=1))
            # 反向传播和优化
            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_size
        
        avg_loss = epoch_loss / num_samples
        print(f"Epoch {epoch+1}, Average MLE Loss: {avg_loss:.4f}")
        print(ground_true_prefer)
        print(torch.softmax(theta,dim=0))
        eval_loss=[]
        with torch.no_grad():
            for batch_obs, batch_actions in test_dataloader:
                batch_size = batch_obs.size(0)
                action_probs_list=[]
                for k in range(batch_size):
                    obs=encoder.decode_observation(batch_obs[k].numpy()) 
                    agent.reset_belief()
                    print(f"  Observation: {obs}")
                    action, action_probs = agent.infer_basedon_prefer(obs, torch.softmax(theta,dim=0)) #ground_true_prefer)#
                    #print(action_probs,batch_actions[k])
                    action_probs=action_probs.unsqueeze(0)
                    action_probs_list.append(action_probs)
                # 计算ELBO损失
                action_probs=torch.cat(action_probs_list, dim=0)
                loss= criterion(action_probs,torch.argmax(batch_actions,dim=1))+regularization_strength * torch.norm(theta)
                eval_loss.append(loss)
            el=np.array(eval_loss)
            print(f"eval loss:{np.mean(el)}")