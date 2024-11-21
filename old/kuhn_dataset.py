import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import numpy as np
from kuhn import KuhnPokerEnv
from data_gen import generate_state_transition_data
import os
# 状态和动作编码
card_encoder = LabelEncoder()
card_encoder.fit(["J", "Q", "K"])
action_encoder = LabelEncoder()
action_encoder.fit(["check", "bet", "call", "fold"])
MASK = 'MASK'


def generate_observations(state) :
    """
    根据当前状态生成双方玩家的观测。
    
    Args:
        state (State): 当前游戏状态，格式为 (hands, actions, pot)
        
    Returns:
        Tuple[Observation, Observation]: 玩家1和玩家2的观测
    """
    hands, actions, pot = state
    if len(hands) != 2:
        raise ValueError("当前实现仅支持2个玩家的Kuhn Poker")
    
    # 玩家1观察到自己的手牌，掩盖玩家2的手牌
    player1_obs = ([hands[0], MASK], actions, pot)
    
    # 玩家2观察到自己的手牌，掩盖玩家1的手牌
    player2_obs = ([MASK, hands[1]], actions, pot)
    
    return player1_obs, player2_obs




class KuhnPokerEncoder:
    def __init__(self):
        # 定义牌的映射
        self.cards = ['J', 'Q', 'K']
        self.card_to_int = {card: idx for idx, card in enumerate(self.cards)}
        self.int_to_card = {idx: card for card, idx in self.card_to_int.items()}
        self.num_cards = len(self.cards)
        
        # 定义动作的映射
        self.actions = ['check','bet', 'call', 'fold', 'none']  # 包含填充动作
        self.action_to_int = {action: idx for idx, action in enumerate(self.actions)}
        self.int_to_action = {idx: action for action, idx in self.action_to_int.items()}
        self.num_actions = len(self.actions)
        
        # 定义最大动作历史长度
        self.max_history = 3  # 根据游戏规则调整
        
        # 定义底池大小的最大值
        self.max_pot_size = 4  # 根据游戏规则调整

    def encode_hand(self, hands):
        """
        编码玩家和对手的手牌
        """
        encoded = []
        for card in hands:
            hand_encoded = [0] * self.num_cards
            if card in self.card_to_int:
                hand_encoded[self.card_to_int[card]] = 1
            else:
                raise ValueError(f"Unknown card: {card}")
            encoded.extend(hand_encoded)
        return encoded  # 长度为 2 * num_cards

    def decode_hand(self, encoded_hand):
        """
        解码玩家和对手的手牌
        """
        hands = []
        for i in range(0, 2 * self.num_cards, self.num_cards):
            hand_slice = encoded_hand[i:i + self.num_cards]
            if sum(hand_slice) != 1:
                raise ValueError(f"Invalid one-hot encoding for hand: {hand_slice}")
            card_index = hand_slice.index(1)
            card = self.int_to_card.get(card_index, None)
            if card is None:
                raise ValueError(f"Invalid card index: {card_index}")
            hands.append(card)
        return hands  # 长度为 2

    def encode_action_history(self, action_history):
        """
        编码动作历史，固定长度，使用 'none' 填充
        """
        encoded = []
        for i in range(self.max_history):
            if i < len(action_history):
                action = action_history[i]
            else:
                action = 'none'
            action_encoded = [0] * self.num_actions
            if action in self.action_to_int:
                action_encoded[self.action_to_int[action]] = 1
            else:
                raise ValueError(f"Unknown action: {action}")
            encoded.extend(action_encoded)
        return encoded  # 长度为 max_history * num_actions

    def decode_action_history(self, encoded_actions):
        """
        解码动作历史
        """
        action_history = []
        for i in range(0, self.max_history * self.num_actions, self.num_actions):
            action_slice = encoded_actions[i:i + self.num_actions]
            if sum(action_slice) != 1:
                raise ValueError(f"Invalid one-hot encoding for action: {action_slice}")
            action_index = action_slice.index(1)
            action = self.int_to_action.get(action_index, None)
            if action is None:
                raise ValueError(f"Invalid action index: {action_index}")
            if action != 'none':
                action_history.append(action)
        return action_history  # 可能长度小于等于 max_history

    def encode_pot_size(self, pot_size):
        """
        编码底池大小，使用独热编码
        """
        encoded = [0] * (self.max_pot_size + 1)  # 包含 0 到 max_pot_size
        if 0 <= pot_size <= self.max_pot_size:
            encoded[pot_size] = 1
        else:
            raise ValueError(f"Pot size out of bounds: {pot_size}")
        return encoded  # 长度为 max_pot_size + 1

    def decode_pot_size(self, encoded_pot):
        """
        解码底池大小
        """
        if sum(encoded_pot) != 1:
            raise ValueError(f"Invalid one-hot encoding for pot size: {encoded_pot}")
        pot_index = encoded_pot.index(1)
        if 0 <= pot_index <= self.max_pot_size:
            return pot_index
        else:
            raise ValueError(f"Invalid pot size index: {pot_index}")

    def encode_state(self, state):
        """
        编码整个状态
        参数:
            state: tuple, 格式为 (player_hands, action_history, pot_size)
                - player_hands: list of two cards, e.g., ['J', 'K']
                - action_history: list of actions, e.g., ['bet', 'fold']
                - pot_size: integer, e.g., 3
        返回:
            encoded_state: list of integers，独热编码后的状态向量
        """
        player_hands, action_history, pot_size = state
        
        # 编码手牌
        hands_encoded = self.encode_hand(player_hands)
        
        # 编码动作历史
        actions_encoded = self.encode_action_history(action_history)
        
        # 编码底池大小
        pot_encoded = self.encode_pot_size(pot_size)
        
        # 组合所有编码
        encoded_state = hands_encoded + actions_encoded + pot_encoded
        return encoded_state

    def decode_state(self, encoded_state):
        """
        解码整个状态
        参数:
            encoded_state: list of integers，独热编码后的状态向量
        返回:
            state: tuple, 格式为 (player_hands, action_history, pot_size)
                - player_hands: list of two cards, e.g., ['J', 'K']
                - action_history: list of actions, e.g., ['bet', 'fold']
                - pot_size: integer, e.g., 3
        """
        expected_length = 2 * self.num_cards + self.max_history * self.num_actions + (self.max_pot_size + 1)
        if len(encoded_state) != expected_length:
            raise ValueError(f"Invalid encoded state length: {len(encoded_state)}. Expected: {expected_length}")
        
        # 解码手牌
        hands_encoded = encoded_state[:2 * self.num_cards]
        player_hands = self.decode_hand(hands_encoded)
        
        # 解码动作历史
        actions_encoded = encoded_state[2 * self.num_cards:2 * self.num_cards + self.max_history * self.num_actions]
        action_history = self.decode_action_history(actions_encoded)
        
        # 解码底池大小
        pot_encoded = encoded_state[2 * self.num_cards + self.max_history * self.num_actions:]
        pot_size = self.decode_pot_size(pot_encoded)
        
        return (player_hands, action_history, pot_size)


class ObservationEncoder(KuhnPokerEncoder):
    """
    对观测进行编码的类，将符号特征转换为数值表示。
    """
    def __init__(self, ):
        self.cards = ['J', 'Q', 'K','MASK']
        self.card_to_int = {card: idx for idx, card in enumerate(self.cards)}
        self.int_to_card = {idx: card for card, idx in self.card_to_int.items()}
        self.num_cards = len(self.cards)
        
        # 定义动作的映射
        self.actions = ['check','bet', 'call', 'fold', 'none']  # 包含填充动作
        self.action_to_int = {action: idx for idx, action in enumerate(self.actions)}
        self.int_to_action = {idx: action for action, idx in self.action_to_int.items()}
        self.num_actions = len(self.actions)
        
        # 定义最大动作历史长度
        self.max_history = 3  # 根据游戏规则调整
        
        # 定义底池大小的最大值
        self.max_pot_size = 4  # 根据游戏规则调整

    def encode_observation(self, obs) -> torch.Tensor:
        """
        将观测编码为数值张量。
        
        Args:
            obs (Observation): 观测，格式为 ([card1, card2], actions, pot)
            
        Returns:
            torch.Tensor: 编码后的观测张量
        """
        cards, actions, pot = obs
        
        # 编码手牌（使用独热编码）
        card_encoding = [0] * self.n_cards
        for card in cards:
            if card in self.card_to_idx:
                card_encoding[self.card_to_idx[card]] = 1
            else:
                raise ValueError(f"未知的手牌: {card}")
        
        # 编码动作（使用独热编码或其他方式，这里示例使用独热平均）
        # 可以根据具体需求调整编码方式
        action_encoding = [0] * self.n_actions
        for action in actions:
            if action in self.action_to_idx:
                action_encoding[self.action_to_idx[action]] = 1
            else:
                raise ValueError(f"未知的动作: {action}")
        # 例如，对动作进行独热编码的平均
        # 也可以使用其他聚合方法，如多个独热编码的拼接
        # 这里为了简化，取平均
        if len(actions) > 0:
            action_encoding = [x / len(actions) for x in action_encoding]
        
        # 编码数值特征（pot）
        pot_encoding = [pot]
        
        # 合并所有编码
        encoded = card_encoding + action_encoding + pot_encoding
        return torch.tensor(encoded, dtype=torch.float32)


encoder=KuhnPokerEncoder()
# 数据集类
class KuhnPokerDataset(Dataset):
    def __init__(self, data):
        self.X = []
        self.y = []
        self.encoder = encoder
        for current_state, action, next_state in data:
            encoded_state, encoded_action = self.encode_state(current_state, action)
            encoded_next_state, _ = self.encode_state(next_state, None)
            self.X.append(torch.cat((torch.tensor(encoded_state, dtype=torch.float32), torch.tensor(encoded_action, dtype=torch.float32))))
            self.y.append(torch.tensor(encoded_next_state, dtype=torch.float32))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def encode_state(self, state, action):
        encoded_state=self.encoder.encode_state(state)
        
        # 编码动作
        action_encoded = np.zeros(len(action_encoder.classes_))
        if action:
            action_encoded[action_encoder.transform([action])[0]] = 1
        
        return encoded_state, action_encoded

def decode_state(encoded_state):
    decoded_state = encoder.decode_state(encoded_state)
    return decoded_state

def save_dataset(dataset, filename):
    """
    保存 PyTorch 数据集到文件。
    参数:
        dataset (Dataset): 需要保存的数据集。
        filename (str): 保存的文件路径。
    """
    data = [(x.numpy(), y.numpy()) for x, y in dataset]
    torch.save(data, filename)
    print(f"数据集已保存到 {filename}")


if __name__ == "__main__":

    # 创建 PyTorch 数据集
    env = KuhnPokerEnv()
    state_transition_data = generate_state_transition_data(env, num_games=10000)
    dataset = KuhnPokerDataset(state_transition_data)

    # 拆分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 保存训练集和测试集
    os.makedirs("datasets", exist_ok=True)
    save_dataset(train_dataset, "datasets/train_dataset.pt")
    save_dataset(test_dataset, "datasets/test_dataset.pt")# 示例使用
