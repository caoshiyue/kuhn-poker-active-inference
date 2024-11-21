import numpy as np

# 定义网格大小
GRID_ROWS = 2
GRID_COLS = 2
NUM_STATES = GRID_ROWS * GRID_COLS

# 定义状态
states = ['s0', 's1', 's2', 's3']  # s3是目标状态

# 定义动作
actions = ['up', 'down', 'left', 'right']

# 定义观察
observations = ['o0', 'o1', 'o2', 'o3']

# 转移模型 P(s'|s,a)
def get_transition_model(): #!  矩阵B 环境转移模型
    transition_model = {}
    for s in range(NUM_STATES):
        row, col = divmod(s, GRID_COLS)
        transition_model[s] = {}
        for a in actions:
            if a == 'up':
                new_row = max(row - 1, 0)
                new_col = col
            elif a == 'down':
                new_row = min(row + 1, GRID_ROWS - 1)
                new_col = col
            elif a == 'left':
                new_row = row
                new_col = max(col - 1, 0)
            elif a == 'right':
                new_row = row
                new_col = min(col + 1, GRID_COLS - 1)
            s_prime = new_row * GRID_COLS + new_col
            transition_model[s][a] = s_prime
    return transition_model

transition_model = get_transition_model()

# 观察模型 P(o|s)
observation_model = np.zeros((NUM_STATES, len(observations))) ##! 矩阵A s<->o 映射
for s in range(NUM_STATES):
    observation_model[s, s] = 1.0  # o0对应s0等

# 初始隐藏状态 P(s)
initial_hidden_state = np.ones(NUM_STATES) / NUM_STATES  # [0.25, 0.25, 0.25, 0.25] #! 矩阵D 认为的初始隐藏状态 

# 贝叶斯更新 P(s|o) ∝ P(o|s) P(s)
def bayesian_update(prior, observation, observation_model):
    o_idx = observations.index(observation)
    likelihood = observation_model[:, o_idx]
    posterior_unnormalized = likelihood * prior
    posterior = posterior_unnormalized / posterior_unnormalized.sum()
    return posterior

# 预测下一个隐藏状态 P(s'|a) = sum_s P(s'|s,a) P(s)
def predict_posterior(prior, action, transition_model):
    predicted_prior = np.zeros(NUM_STATES)
    for s in range(NUM_STATES):
        s_prime = transition_model[s][action]
        predicted_prior[s_prime] += prior[s]
    # 归一化
    predicted_prior /= predicted_prior.sum()
    return predicted_prior

# 计算自由能 F = KL(Q(s)||P(s|a))
def calculate_free_energy(posterior, predicted_prior):
    # KL散度：KL(Q||P)
    kl_div = np.sum(posterior * np.log(posterior / predicted_prior))
    return kl_div

# 选择最小化自由能的动作
def select_action(current_prior, transition_model, observation_model):
    action_free_energies = {}
    for a in actions:
        # 预测动作后的先验
        predicted_prior = predict_posterior(current_prior, a, transition_model)
        # 预测观察（期望观察）
        expected_observation = np.dot(predicted_prior, observation_model)
        # 假设我们期望达到目标状态 s3，对应观察 o3
        preferred_observation = np.zeros(len(observations)) #! 矩阵C 目标obs
        preferred_observation[3] = 1.0  # 假设目标是o3
        # 计算自由能：KL(Q(o)||P(o|a)) 其中Q(o)是期望观察
        free_energy = np.sum(preferred_observation * np.log(preferred_observation / expected_observation))
        action_free_energies[a] = free_energy
    # 选择自由能最小的动作
    best_action = min(action_free_energies, key=action_free_energies.get)
    return best_action, action_free_energies

# 示例执行
current_prior = initial_hidden_state.copy()

# 选择动作
best_action, free_energies = select_action(current_prior, transition_model, observation_model)
print("各动作的自由能：")
for a, fe in free_energies.items():
    print(f"动作 '{a}': 自由能 = {fe:.4f}")
print(f"选择的最佳动作: '{best_action}'")

# 假设选择了最佳动作并执行，接收观察
action = best_action
predicted_prior = predict_posterior(current_prior, action, transition_model)
print(f"\n采取动作 '{action}' 后，预测的隐藏状态 P(s')：")
print(predicted_prior)

# 假设收到观察 'o1'
observation = 'o1'
posterior = bayesian_update(predicted_prior, observation, observation_model)
print(f"\n收到观察 '{observation}' 后，更新的后验隐藏状态 P(s|o)：")
for s in range(NUM_STATES):
    print(f"P(s={states[s]}|o={observation}) = {posterior[s]:.2f}")
