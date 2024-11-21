import numpy as np
from kuhn import KuhnPokerEnv
import random 
import json
from collections import defaultdict

def generate_state_transition_data(env: KuhnPokerEnv, num_games: int = 10000) :
    """
    生成大量对局数据，记录 (当前状态, 动作, 下一状态)。
    参数:
        env (KuhnPokerEnv): Kuhn Poker 环境。
        num_games (int): 对局数量。
    返回:
        data (List[Tuple]): 包含 (state, action, next_state) 的数据。
    """
    data = []
    
    for _ in range(num_games):
        env.reset()
        done = False
        
        while not done:
            # 获取当前状态（完整状态）
            current_state = (tuple(env.player_hands), tuple(env.action_history[:],), env.pot_size)
            
            # 随机选择一个合法动作
            valid_actions = env.get_valid_actions()
            action = random.choice(valid_actions)
            
            # 执行动作，获取动作后的完整状态
            _, _, done, _ = env.step(action)
            next_state = (tuple(env.player_hands), tuple(env.action_history[:],), env.pot_size)
            
            # 保存 (当前状态, 动作, 下一状态)
            data.append((current_state, action, next_state))
    
    return data

if __name__ == "__main__":
# 生成数据
    env = KuhnPokerEnv()

    state_transition_data = generate_state_transition_data(env, num_games=10000)
    print(f"生成的状态转移数据量: {len(state_transition_data)} 条")
    state_counts = defaultdict(int)
    for data_point in state_transition_data:
        _, _, next_state = data_point
        state_counts[next_state] += 1

    for state, count in state_counts.items():
        print(f"状态 {state} 的样本数量: {count}")


    # unique_data = set(json.dumps(item, sort_keys=True) for item in state_transition_data)
    # unique_states = set(json.dumps(item[2], sort_keys=True) for item in state_transition_data)
    # print(unique_states)

