##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2024-11-20 22:33:49
## 

import itertools
import numpy as np

# 定义游戏元素
cards = ['J', 'Q', 'K']
actions = ['check', 'bet', 'call', 'fold']
MASK = "MASK"
def one_hot_encode(index, size):
    vector = np.zeros(size)
    vector[index] = 1
    return vector

def one_hot_decode(one_hot_vector):
    # 确保只有一个 '1'

    if not isinstance(one_hot_vector, np.ndarray):
        one_hot_vector=np.array(one_hot_vector)
    indices = np.where(one_hot_vector == 1)[0]
    if len(indices) == 0:
        raise ValueError("No '1' found in the one-hot vector.")
    if len(indices) > 1:
        raise ValueError("Multiple '1's found in the one-hot vector.")
    return indices[0]

# 可选：创建编码器类以便更加模块化
class KuhnPokerEncoder:
    def __init__(self):
        self.cards = ['J', 'Q', 'K']
        self.actions = ['check', 'bet', 'call', 'fold']
        self.hand_assignments = list(itertools.permutations(self.cards, 2))
        self.action_histories = [
            (),
            ('check',),
            ('bet',),
            ('check', 'check'),
            ('check', 'bet'),
            ('bet', 'fold'),
            ('bet', 'call'),
            ('check', 'bet', 'fold'),
            ('check', 'bet', 'call'),
        ]
        self.states = []
        for hand in self.hand_assignments:
            for history in self.action_histories:
                pot = self.calculate_pot(history)
                self.states.append((hand, tuple(history), pot))
        self.state_to_index = {state: idx for idx, state in enumerate(self.states)}
        self.num_states = len(self.states)
        self.index_to_state = {idx: state for state, idx in self.state_to_index.items()}

        
        # Generate observations
        observations = []
        for state in self.states:
            hand, history, pot = state
            player1_obs = ((hand[0], MASK), tuple(history), pot)
            player2_obs = ((MASK, hand[1]), tuple(history), pot)
            observations.append(player1_obs)
            observations.append(player2_obs)

        seen = set()
        self.observations = []
        for item in observations:
            if item not in seen:
                self.observations.append(item)
                seen.add(item)

        self.observation_to_index = {obs: idx for idx, obs in enumerate(self.observations)}
        self.num_observations = len(self.observations)
        self.index_to_observation = {idx: obs for obs, idx in self.observation_to_index.items()}

        self.action_to_index = {action: idx for idx, action in enumerate(self.actions)}
        self.index_to_action = {idx: action for action, idx in self.action_to_index.items()}
        self.num_actions = len(self.actions)

    
    def calculate_pot(self, history):
        pot = 2
        for action in history:
            if action in ['bet', 'call']:
                pot += 1
        return pot
    
    def encode_state(self, state):
        index = self.state_to_index.get(state)
        if index is None:
            raise ValueError(f"State {state} not found in state space.")
        return one_hot_encode(index, self.num_states)
    
    def encode_observation(self, observation):
        index = self.observation_to_index.get(observation)
        if index is None:
            raise ValueError(f"Observation {observation} not found in observation space.")
        return one_hot_encode(index, self.num_observations)
    
    def get_state_index(self, state):
        return self.state_to_index.get(state)
    
    def get_observation_index(self, observation):
        return self.observation_to_index.get(observation)
    
    def decode_state(self, one_hot_vector):
        index = one_hot_decode(one_hot_vector)
        state = self.index_to_state.get(index)
        if state is None:
            raise ValueError(f"Index {index} not found in state space.")
        return state

    def decode_observation(self, one_hot_vector):
        index = one_hot_decode(one_hot_vector)
        observation = self.index_to_observation.get(index)
        if observation is None:
            raise ValueError(f"Index {index} not found in observation space.")
        return observation
    def get_state_index(self, state):
        return self.state_to_index.get(state)

    def get_observation_index(self, observation):
        return self.observation_to_index.get(observation)
    
    def encode_action(self, action):
        """
        编码动作，将动作字符串转换为整数索引。
        """
        index = self.action_to_index.get(action)
        if index is None:
            raise ValueError(f"Action '{action}' not recognized.")
        action_one_hot = one_hot_encode(index, self.num_actions) 
        return action_one_hot

    def decode_action(self, action_one_hot):
        """
        解码动作，将整数索引转换为动作字符串。
        """
        index = one_hot_decode(action_one_hot)
        if index is None:
            raise ValueError(f"Action index '{index}' is invalid.")
        action=self.index_to_action.get(index)
        return action

# 使用编码器类
if __name__ == "__main__":
    encoder = KuhnPokerEncoder()

    example_state = (('J', 'K'), ('bet', 'fold'), 3)
    state_one_hot = encoder.encode_state(example_state)
    print(f"One-hot encoding for state {example_state}:\n{state_one_hot}")

    example_observation = (('J', 'MASK'), ('bet', 'fold'), 3)
    observation_one_hot = encoder.encode_observation(example_observation)
    print(f"One-hot encoding for observation {example_observation}:\n{observation_one_hot}")

    # 解码示例：从 one-hot 向量恢复状态和观测
    decoded_state = encoder.decode_state(state_one_hot)
    print(f"Decoded state from one-hot:\n{decoded_state}")

    decoded_observation = encoder.decode_observation(observation_one_hot)
    print(f"Decoded observation from one-hot:\n{decoded_observation}")

    # 编码另一个状态并解码
    another_state = (('Q', 'J'), ('check', 'bet'), 3)
    another_state_one_hot = encoder.encode_state(another_state)
    print(f"One-hot encoding for state {another_state}:\n{another_state_one_hot}")

    another_observation = (('Q', 'MASK'), ('check', 'bet'), 3)
    another_observation_one_hot = encoder.encode_observation(another_observation)
    print(f"One-hot encoding for observation {another_observation}:\n{another_observation_one_hot}")

    decoded_another_state = encoder.decode_state(another_state_one_hot)
    print(f"Decoded state from one-hot:\n{decoded_another_state}")

    decoded_another_observation = encoder.decode_observation(another_observation_one_hot)
    print(f"Decoded observation from one-hot:\n{decoded_another_observation}")
    actions = ['check', 'bet', 'call', 'fold', 'invalid_action']

    print("动作编码示例:")
    for action in actions:
        try:
            encoded = encoder.encode_action(action)
            print(f"Action '{action}' encoded as {encoded}")
        except ValueError as e:
            print(e)

    print("\n动作解码示例:")
    idx = encoded

    try:
        decoded = encoder.decode_action(idx)
        print(f"Index '{idx}' decoded as '{decoded}'")
    except ValueError as e:
        print(e)