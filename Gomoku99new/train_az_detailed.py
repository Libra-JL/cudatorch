import matplotlib

matplotlib.use('TkAgg')
import torch, torch.optim as optim, torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from collections import deque
import random, os, yaml, traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
from game import GomokuGame
from ai_pytorch_resnet_v2 import Net  # 确保导入的是ResNet版本


def load_config(path="config.yaml"):
    with open(path, 'r',encoding='utf-8') as f: return yaml.safe_load(f)


# (LivePlotter, MCTSNode, run_mcts, self_play 保持不变)
# ...
class LivePlotter:
    def __init__(self, title, xlabel, ylabel):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.lines = {}
        self.x_data = []
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)

    def update(self, iteration, metrics: dict):
        if not self.x_data or iteration > self.x_data[-1]: self.x_data.append(iteration)
        for label, value in metrics.items():
            if value is None: continue
            if label not in self.lines: self.lines[label] = {"y_data": [], "line_obj":
                self.ax.plot([], [], marker='o', markersize=4, linestyle='-', label=label)[0]}
            y_len = len(self.lines[label]["y_data"])
            if y_len < len(self.x_data):
                self.lines[label]["y_data"].extend([float('nan')] * (len(self.x_data) - 1 - y_len))
                self.lines[label]["y_data"].append(value)
            else:
                self.lines[label]["y_data"][-1] = value
            self.lines[label]["line_obj"].set_data(self.x_data, self.lines[label]["y_data"])
        self.ax.legend()
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

    def save_and_close(self, filename):
        plt.ioff(); self.fig.savefig(filename); print(f"图形已保存为: {filename}"); plt.show()


class MCTSNode:
    def __init__(self, parent=None, prior_p=1.0):
        self.parent, self.children, self.visit_count, self.q_value, self.prior_p = parent, {}, 0, 0, prior_p

    def select(self, c_puct):
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children: self.children[action] = MCTSNode(parent=self, prior_p=prob)

    def update(self, value):
        self.visit_count += 1; self.q_value += (value - self.q_value) / self.visit_count

    def get_value(self, c_puct):
        u = (c_puct * self.prior_p * np.sqrt(self.parent.visit_count) / (
                    1 + self.visit_count)) if self.visit_count > 0 else c_puct * self.prior_p * np.sqrt(
            self.parent.visit_count)
        return self.q_value + u

    def is_leaf(self):
        return len(self.children) == 0


def run_mcts(game_state, nnet, config, add_noise=False):
    root = MCTSNode()
    for _ in range(config['mcts']['simulations']):
        node = root
        temp_game = game_state.clone()
        while not node.is_leaf(): action, node = node.select(config['mcts']['cpuct']); temp_game.make_move(action[0],
                                                                                                           action[1])
        if not temp_game.game_over:
            with torch.no_grad():
                policy, leaf_value = nnet(temp_game.get_board_tensor(config['device']))
            leaf_value = leaf_value.item()
            policy = torch.exp(policy).squeeze(0).cpu().detach().numpy()
            valid_moves = temp_game.get_valid_moves()
            action_priors = {move: policy[move[0] * 9 + move[1]] for move in valid_moves}
            if add_noise and node == root and valid_moves:
                noise = np.random.dirichlet([config['loss']['dirichlet_alpha']] * len(valid_moves))
                for i, move in enumerate(valid_moves): action_priors[move] = (1 - config['loss']['dirichlet_epsilon']) * \
                                                                             action_priors[move] + config['loss'][
                                                                                 'dirichlet_epsilon'] * noise[i]
            if action_priors: node.expand(action_priors.items())
        else:
            leaf_value = 1.0 if temp_game.winner == game_state.current_player else -1.0 if temp_game.winner != 0 else 0.0
        while node is not None: node.update(-leaf_value); leaf_value = -leaf_value; node = node.parent
    counts = [node.visit_count for _, node in root.children.items()]
    actions = [act for act, _ in root.children.items()]
    return actions, counts


def self_play(nnet, config):
    game = GomokuGame(size=9, win_condition=5)
    game_history = []
    while not game.game_over:
        actions, counts = run_mcts(game, nnet, config, add_noise=True)
        if not actions: break
        policy_target = np.zeros(81, dtype=np.float32)
        for i, action in enumerate(actions): policy_target[action[0] * 9 + action[1]] = counts[i]
        policy_target /= np.sum(policy_target)
        game_history.append([game.get_board_tensor(config['device'], to_numpy=True), policy_target])
        move_probs = np.array(counts, dtype=np.float32) / sum(counts)
        action_to_take = actions[np.random.choice(len(actions), p=move_probs)]
        game.make_move(action_to_take[0], action_to_take[1])
    final_data = []
    if game.winner is not None:
        z = 1.0 if game.winner != 0 else 0.0
        for board_tensor, policy in reversed(game_history):
            final_data.append([board_tensor, policy, z])
            z = -z
    return final_data[::-1]


# --- 【核心修改】损失函数返回分解后的值 ---
def calculate_loss(nnet, data_batch, config):
    boards, policies, values = zip(*data_batch)
    boards_tensor = torch.FloatTensor(np.array(boards)).to(config['device'])
    target_policies = torch.FloatTensor(np.array(policies)).to(config['device'])
    target_values = torch.FloatTensor(np.array(values)).view(-1, 1).to(config['device'])
    pred_policies, pred_values = nnet(boards_tensor)
    policy_loss = -torch.sum(target_policies * pred_policies) / target_policies.size()[0]
    value_loss = F.mse_loss(pred_values, target_values)
    total_loss = policy_loss + config['loss']['value_loss_weight'] * value_loss
    return total_loss, policy_loss, value_loss


def train_with_replay(nnet, replay_buffer, optimizer, config):
    batch_size = config['network']['batch_size']
    if len(replay_buffer) < batch_size * 10: print(
        f"经验池数据量 ({len(replay_buffer)}) 不足..."); return None, None, None
    nnet.train()
    total_loss_sum, policy_loss_sum, value_loss_sum = 0, 0, 0
    TRAINING_STEPS_PER_ITERATION = 200
    for _ in range(TRAINING_STEPS_PER_ITERATION):
        training_batch = random.sample(replay_buffer, batch_size)
        optimizer.zero_grad()
        total_loss, policy_loss, value_loss = calculate_loss(nnet, training_batch, config)
        total_loss.backward()
        optimizer.step()
        total_loss_sum += total_loss.item()
        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()
    n_steps = TRAINING_STEPS_PER_ITERATION
    return total_loss_sum / n_steps, policy_loss_sum / n_steps, value_loss_sum / n_steps


def evaluate_on_dataset(nnet, dataset, config):
    if not dataset: return None, None, None
    nnet.eval()
    with torch.no_grad():
        total_loss_sum, policy_loss_sum, value_loss_sum = 0, 0, 0
        batch_size = config['network']['batch_size']
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        if num_batches == 0: return None, None, None
        for i in range(num_batches):
            batch = dataset[i * batch_size:(i + 1) * batch_size]
            total_loss, policy_loss, value_loss = calculate_loss(nnet, batch, config)
            total_loss_sum += total_loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
    return total_loss_sum / num_batches, policy_loss_sum / num_batches, value_loss_sum / num_batches


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device

    net_config = config['network']
    nnet = Net(
        board_size=9,
        num_res_blocks=net_config['num_res_blocks'],
        num_channels=net_config['num_channels'],
        dropout_rate=net_config['dropout_rate']
    ).to(device)

    optimizer = optim.AdamW(nnet.parameters(), lr=net_config['initial_lr'], weight_decay=net_config['weight_decay'])
    scheduler_config = config['scheduler']
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_config['factor'],
                                  patience=scheduler_config['patience'], min_lr=scheduler_config['min_lr'])
    replay_buffer = deque(maxlen=config['data']['replay_buffer_size'])
    start_iteration = 1

    checkpoint_path = config['checkpoint']['path']
    if os.path.exists(checkpoint_path):
        print(f"正在从检查点加载: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            nnet.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_iteration = checkpoint['iteration'] + 1
            replay_buffer.extend(checkpoint['replay_buffer'])
            print(f"成功加载，将从第 {start_iteration} 次迭代继续训练。")
        except Exception as e:
            print(f"加载检查点失败: {e}。将从头开始训练。")
    else:
        print("未找到检查点，从头开始训练。")

    total_loss_plotter = LivePlotter(title='Total Loss (Train/Val/Test)', xlabel='Iteration', ylabel='Loss')
    detailed_loss_plotter = LivePlotter(title='Policy vs. Value Loss (Validation)', xlabel='Iteration',
                                        ylabel='Loss Breakdown')

    try:
        for i in range(start_iteration, config['training']['max_iterations'] + 1):
            print(f"--- 迭代 {i}/{config['training']['max_iterations']} ---")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr}")
            nnet.eval()
            iteration_data = []
            print("开始自我对弈...")
            for _ in tqdm(range(config['training']['num_games_per_iteration']), desc=f"迭代 {i} 自我对弈"):
                game_data = self_play(nnet, config)
                iteration_data.extend(game_data)
            if not iteration_data: print("本轮未产生有效数据，跳过。"); continue
            replay_buffer.extend(iteration_data)

            random.shuffle(iteration_data)
            test_split_point = int(len(iteration_data) * config['data']['test_split'])
            val_split_point = test_split_point + int(len(iteration_data) * config['data']['validation_split'])
            test_set = iteration_data[:test_split_point]
            validation_set = iteration_data[test_split_point:val_split_point]

            print(f"开始训练网络... (经验池大小: {len(replay_buffer)})")
            train_total, train_policy, train_value = train_with_replay(nnet, list(replay_buffer), optimizer, config)
            val_total, val_policy, val_value = evaluate_on_dataset(nnet, validation_set, config)
            test_total, test_policy, test_value = evaluate_on_dataset(nnet, test_set, config)

            print("损失分解:")
            print(
                f"  - 训练集: 总={train_total or 'N/A':.4f}, 策略={train_policy or 'N/A':.4f}, 价值={train_value or 'N/A':.4f}")
            print(
                f"  - 验证集: 总={val_total or 'N/A':.4f}, 策略={val_policy or 'N/A':.4f}, 价值={val_value or 'N/A':.4f}")

            total_loss_plotter.update(i, {"Train Total": train_total, "Val Total": val_total, "Test Total": test_total})
            detailed_loss_plotter.update(i, {"Val Policy": val_policy, "Val Value": val_value})

            if val_total: scheduler.step(val_total)

            torch.save(
                {'iteration': i, 'model_state_dict': nnet.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(), 'replay_buffer': list(replay_buffer)}, checkpoint_path)
            print(f"检查点已保存到: {checkpoint_path}\n")
    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\n训练被手动中断。")
        else:
            print(f"\n发生错误: {e}"); traceback.print_exc()
    finally:
        print("正在保存最终的收敛曲线图...")
        total_loss_plotter.save_and_close("total_loss_curves_final.png")
        detailed_loss_plotter.save_and_close("detailed_loss_curves_final.png")



if __name__ == "__main__":
    main()