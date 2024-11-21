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

encoder = KuhnPokerEncoder()
# 加载训练集和测试集
train_data = load_dataset("datasets/train_dataset.pt")
test_data = load_dataset("datasets/test_dataset.pt")

# 转换为 DataLoader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")

# 模型实例化
input_dim = len(train_data[0][0])  # 输入维度 = 状态编码 + 动作编码
output_dim = len(train_data[0][1])  # 输出维度 = 下一状态编码
model = TransitionModel(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train_model(model, train_loader, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.8f}")

train_model(model, train_loader)

# 验证模型
def evaluate_model(model, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_loader):.8f}")

evaluate_model(model, test_loader)

def binarize_output(encoded_state, threshold=0.5):
    """
    将连续的编码输出转换为独热编码向量
    """
    return (encoded_state >= threshold).float()

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

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # 模型预测
            predictions = model(X_batch)
            
            stateinput=encoder.decode_state(X_batch[0][:-4])
            action=encoder.decode_action(X_batch[0][-4:])
            #print(stateinput,action)
            # 解码预测状态和真值状态
            for raw_pred, true in zip(predictions, y_batch):
                pred=binarize_output(raw_pred)
                true = true.cpu().tolist()
                pred = pred.cpu().tolist()

                decoded_true = encoder.decode_state(true)
                
                decoded_pred = encoder.decode_state(pred)
                #print(decoded_pred)
                # 比较解码后的状态
                if decoded_pred == decoded_true:
                    correct_predictions += 1
                else:
                    print(decoded_pred,decoded_true)
                total_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"准确率: {accuracy:.2%}")
    return accuracy
    
accuracy = compute_accuracy(model, test_loader)
torch.save(model, 'state_transition_model')
