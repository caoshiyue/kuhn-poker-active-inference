import random
from typing import List, Tuple

class KuhnPokerEnv:
    def __init__(self):
        # 初始化变量
        self.cards = ["J", "Q", "K"]  # 游戏卡牌
        self.reset()
        
    def reset(self):
        """
        重置环境，随机分配卡牌，清空历史记录和底池。
        """
        # 随机分配卡牌
        random.shuffle(self.cards)
        self.player_hands = [self.cards[0], self.cards[1]]  # 玩家 A 和 B 的手牌
        self.discard_card = self.cards[2]  # 丢弃的卡牌
        
        # 初始化游戏状态
        self.action_history = []  # 行动历史记录
        self.pot_size = 2  # 初始底池大小
        self.current_player = 0  # 玩家 A 先行动（0 表示玩家 A，1 表示玩家 B）
        self.done = False  # 游戏是否结束
        self.winner = None  # 胜利者
        
        # 返回初始观测
        return self.get_observation()

    def get_observation(self) -> Tuple[str, List[str], int]:
        """
        获取当前玩家的观测信息。
        """
        own_hand = ["MASK","MASK"]
        own_hand[self.current_player] = self.player_hands[self.current_player]
        return tuple(own_hand),  tuple(self.action_history,), self.pot_size

    def get_agent_observation(self,player) -> Tuple[str, List[str], int]:
        """
        获取当前玩家的观测信息。
        """
        own_hand = ["MASK","MASK"]
        own_hand[player] = self.player_hands[player]
        return tuple(own_hand),  tuple(self.action_history,), self.pot_size

    def step(self, action: str):
        """
        执行当前玩家的动作。
        参数:
            action (str): 当前玩家的动作，可以是 'check', 'bet', 'call', 'fold'。
        返回:
            observation (Tuple): 下一步的观测。
            reward (int): 当前玩家获得的奖励。
            done (bool): 游戏是否结束。
            info (dict): 额外信息。
        """
        # 验证动作是否合法
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            raise ValueError(f"非法动作: {action}. 可用动作: {valid_actions}")

        # 处理玩家动作
        self.action_history.append(action)
        if action == "bet":
            self.pot_size += 1  # 增加底池
        elif action == "call":
            self.pot_size += 1  # 增加底池
            self.done = True  # 游戏结束，双方亮牌
            self.determine_winner()
        elif action == "fold":
            self.done = True  # 游戏结束，对手获胜
            self.winner = 1 - self.current_player
        elif action == "check" and len(self.action_history) > 1 and self.action_history[-2] == "check":
            self.done = True  # 双方都选择 check，游戏结束
            self.determine_winner()
        
        # 切换玩家
        if not self.done:
            self.current_player = 1 - self.current_player
        
        # 返回结果
        observation = self.get_observation()
        reward = self.get_reward()
        info = {"winner": self.winner, "action_history": self.action_history}
        return observation, reward, self.done, info

    def get_valid_actions(self) -> List[str]:
        """
        返回当前玩家的合法动作。
        """
        if not self.action_history:
            return ["check", "bet"]
        last_action = self.action_history[-1]
        if last_action == "check":
            return ["check", "bet"]
        elif last_action == "bet":
            return ["call", "fold"]
        return []

    def determine_winner(self):
        """
        根据手牌确定胜利者。
        """
        if not self.done :
            return
        if self.winner is not None:
            return self.winner
        hand_values = {"J": 1, "Q": 2, "K": 3}
        player_a_hand = hand_values[self.player_hands[0]]
        player_b_hand = hand_values[self.player_hands[1]]
        
        if player_a_hand > player_b_hand:
            self.winner = 0  # 玩家 A 胜
        else:
            self.winner = 1  # 玩家 B 胜
        return self.winner
        
    def get_reward(self) -> int:
        """
        返回当前玩家的奖励。奖励为底池中的筹码数（胜者）或 0（败者）。
        """
        if not self.done:
            return 0
        return self.pot_size #if self.current_player == self.winner else 0
