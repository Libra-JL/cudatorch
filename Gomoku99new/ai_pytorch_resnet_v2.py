import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random

import yaml


# --- 定义一个残差块 ---
class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.leaky_relu(out, 0.1)
        return out


# --- 构建一个基于残差块的ResNet ---
class Net(nn.Module):
    def __init__(self, board_size=9, num_res_blocks=5, num_channels=128, dropout_rate=0.3):
        super(Net, self).__init__()
        self.board_size = board_size
        self.dropout_rate = dropout_rate

        self.initial_conv = nn.Conv2d(2, num_channels, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_channels)

        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])

        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, board_size * board_size)

        self.value_conv = nn.Conv2d(num_channels, 16, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.leaky_relu(self.initial_bn(self.initial_conv(x)), 0.1)
        for block in self.res_blocks:
            x = block(x)

        policy = F.leaky_relu(self.policy_bn(self.policy_conv(x)), 0.1)
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)

        value = F.leaky_relu(self.value_bn(self.value_conv(x)), 0.1)
        value = value.view(value.size(0), -1)
        value = F.leaky_relu(self.value_fc1(value))
        value = F.dropout(value, p=self.dropout_rate, training=self.training)
        value = torch.tanh(self.value_fc2(value))

        return policy, value


# --- 这是专门用于“使用”模型的AI驱动程序 ---
class GomokuAI_PyTorch:
    def __init__(self, board_size=9, model_path="az_checkpoint.pth", config_path="config.yaml"):
        self.board_size = board_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 从config加载网络参数
        with open(config_path, 'r',encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.model = Net(
            board_size=board_size,
            num_res_blocks=config['network']['num_res_blocks'],
            num_channels=config['network']['num_channels'],
            dropout_rate=config['network']['dropout_rate']
        ).to(self.device)

        self.model_loaded = False
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device,weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                self.model_loaded = True
                print(f"成功加载模型 {model_path} 到 {self.device}！AI已准备就绪。")
            except Exception as e:
                print(f"加载模型失败: {e}\nAI将使用随机算法。")
        else:
            print(f"未找到检查点文件 {model_path}。AI将使用随机算法。")

    def _preprocess_board(self, board, current_player):
        board_tensor = torch.zeros(1, 2, self.board_size, self.board_size, device=self.device)
        board_tensor[0, 0][board == current_player] = 1
        board_tensor[0, 1][board == (3 - current_player)] = 1
        return board_tensor


    def find_best_move(self, game_instance):
        valid_moves = game_instance.get_valid_moves()
        if not valid_moves: return None
        if self.model_loaded:
            board_tensor = self._preprocess_board(game_instance.board, game_instance.current_player)
            with torch.no_grad():
                log_policy, _ = self.model(board_tensor)
            policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
            valid_policy = {move: policy[move[0] * self.board_size + move[1]] for move in valid_moves}
            if valid_policy:
                return max(valid_policy, key=valid_policy.get)
            else:
                return random.choice(valid_moves)
        else:
            return random.choice(valid_moves)