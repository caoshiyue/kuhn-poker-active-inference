##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2024-11-21 16:50:19
## 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from kuhn_encode import KuhnPokerEncoder
from models import *

def load_dataset(filename):
    data = torch.load(filename)
    return [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for x, y in data]

def focal_loss(output, target, alpha=0.25, gamma=2.0):
    # output: 模型输出的概率 (sigmoid 之后)
    # target: 真实标签 (0 或 1)
    pt = torch.where(target == 1, output, 1 - output)  # 获取 pt
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt + 1e-12)  # 避免 log(0)
    return loss.mean()

encoder = KuhnPokerEncoder()
# 加载训练集和测试集
train_data = load_dataset("datasets/obs_train_dataset.pt")
test_data = load_dataset("datasets/obs_test_dataset.pt")

# 转换为 DataLoader
train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")

# 模型实例化
input_dim = len(train_data[0][0])  # 输入维度 = 状态编码 + 动作编码
output_dim = len(train_data[0][1])  # 输出维度 = 下一状态编码
model = ObservationToStateModel(input_dim, output_size=output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 验证模型
def evaluate_model(model, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            targets_indices = torch.argmax(y_batch, dim=1)
            loss = criterion(predictions, targets_indices)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_loader):.8f}")

def label_smoothing(target, epsilon=0.1, num_classes=54):
    # target: 真实标签 (one-hot 编码)
    smooth_target = target * (1 - epsilon) + epsilon / num_classes
    return smooth_target

# 训练模型
def train_model(model, train_loader, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 前向传播
            outputs = model(inputs)
            
            # 将目标状态编码从 one-hot 转换为类别索引
            targets = torch.argmax(targets, dim=1)
            #targets=label_smoothing(targets)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print(outputs,targets)
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        #evaluate_model(model, test_loader)

train_model(model, train_loader)



def binarize_output(encoded_state, threshold=0.5):
    """
    将连续的编码输出转换为独热编码向量
    """
    return (encoded_state >= threshold).float()


def get_prob_state(output_probs_np):
    threshold = 0.25
    for idx, prob in enumerate(output_probs_np):
        if prob >= threshold:
            state = encoder.index_to_state.get(idx, "未知状态")
            print(f"高置信度预测: 状态 {state}，概率 = {prob:.4f}")

def get_topk_state(predictions):
    top_k_output = torch.topk(predictions, 3, dim=1).indices.tolist()
    for  k in top_k_output[0]:
        state = encoder.index_to_state.get(k, "未知状态")
        print(f"高置信度预测: 状态 {state}，概率 = {predictions[0][k]:.6f}")



def compute_accuracy(model, test_loader):
    """
    计算模型在测试集上的预测准确率。
    参数:
        model (nn.Module): 训练好的状态转移模型。
        test_loader (DataLoader): 测试数据加载器。
    返回:
        准确率 (float): 模型预测准确率。
    """
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    temperature = 0.1
    with torch.no_grad():
        i=0
        for  X_batch, y_batch in test_loader:
            # 模型预测
            i+=1
            predictions = model(X_batch)/temperature
            stateinput=encoder.decode_observation(X_batch[0])
            stateoutput=encoder.decode_state(y_batch[0])
            print(f"输入观测： {stateinput}; 对应状态： {stateoutput}")
            predictions=torch.softmax(predictions, dim=1)
            output_probs_np = predictions.numpy().flatten()
            get_topk_state(predictions)

            # print(output_probs_np)
            # get_prob_state(output_probs_np)
            if i>100:
                break

compute_accuracy(model, test_loader)
torch.save(model, 'obs_model')