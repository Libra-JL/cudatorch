import numpy as np
import torch

class GomokuGame:
    """
    五子棋游戏逻辑类
    """

    def __init__(self, size=9, win_condition=5):
        self.size = size
        self.win_condition = win_condition
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # 1 代表玩家 (黑棋), 2 代表 AI (白棋)
        self.game_over = False
        self.winner = None

        # 新增 history 属性
        # 用一个列表来记录每一步的落子动作 (row, col)
        self.history = []

    def reset(self):
        """重置游戏"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        # 重置时也要清空历史记录
        self.history = []

    def make_move(self, row, col):
        """
        在指定位置落子
        返回 True 如果落子成功, False 如果位置已被占用或游戏已结束
        """
        if self.game_over or self.board[row, col] != 0:
            return False

        self.board[row, col] = self.current_player

        # 每次成功落子后，将动作记录到历史中
        self.history.append((row, col))

        if self.check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        elif self.is_board_full():
            self.game_over = True
            self.winner = 0  # 0 代表平局

        # 切换玩家
        self.current_player = 3 - self.current_player  # 1 -> 2, 2 -> 1
        return True

    def check_win(self, row, col):
        """检查在最后一次落子后是否有玩家获胜"""
        player = self.board[row, col]
        if player == 0:
            return False

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, self.win_condition):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            for i in range(1, self.win_condition):
                r, c = row - i * dr, col - i * dc
                if 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            if count >= self.win_condition:
                return True
        return False


    def is_board_full(self):
        """检查棋盘是否已满"""
        return np.all(self.board != 0)

    def get_valid_moves(self):
        """获取所有可以落子的空位"""
        return list(zip(*np.where(self.board == 0)))

    def get_board_tensor(self, device, to_numpy=False):
        board_tensor = torch.zeros(1, 2, 9, 9)
        player_perspective_board = np.copy(self.board)
        player_perspective_board[player_perspective_board == (3 - self.current_player)] = -1
        player_perspective_board[player_perspective_board == self.current_player] = 1
        board_tensor[0, 0][player_perspective_board == 1] = 1
        board_tensor[0, 1][player_perspective_board == -1] = 1
        if to_numpy: return board_tensor.squeeze(0).cpu().detach().numpy()
        return board_tensor.to(device)

    def clone(self):
        new_game = GomokuGame(9, 5)
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.history = list(self.history)
        return new_game