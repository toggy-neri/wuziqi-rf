


from datetime import datetime

import numpy as np
from torch.nn import functional as F
import torch

from mcts import MCTS
from experience_replay import ReplayMemory
from wuziqi_env import WuziqiEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class pretrainer:
    def __init__(self, agent):
        self.agent = agent
        self.board_size = agent.board_size
        self.network = agent.network
    def pretrain(self,pkl_path: str, memory: ReplayMemory, steps: int = 5000, batch_size: int = 64, label_smoothing: float = 0.1, val_ratio: float = 0.2):
            import pickle
            with open(pkl_path, 'rb') as f:
                records = pickle.load(f)
            
            print(f"总样本数: {len(records)}, 开始重建env...")
            current_time = datetime.now()
            start_time = current_time
            pretrain_env = WuziqiEnv(board_size= self.board_size)
            pretrain_env.reset()
            
            valid_count = 0
            prev_move_count = 0
            all_samples = []  # 先收集所有样本再划分

            last_game_id = -1  # 初始化一个不可能的 game_id

            for idx, r in enumerate(records):
                board = r['board']
                next_moves = r['next_moves']
                value = r['value']
                current_player = r['current_player']
                game_id = r['game_id']
                
                # ================= 核心修复：根据 game_id 重置环境 =================
                if game_id != last_game_id:
                    # 检测到新的对局开始，重置环境
                    pretrain_env.reset()
                    last_game_id = game_id  # 更新追踪的 game_id
                # ================================================================

                next_moves = np.array(next_moves, dtype=np.float32)
                
                # 获取当前最佳走步 (用于同步 env 状态)
                top5_indices = np.argsort(next_moves)[-5:][::-1]
                next_move = int(top5_indices[0])  # 确保转为 python int
                
                # 先获取当前状态的特征！非常重要！
                # 必须在 step 之前获取，因为 get_channel_state 需要的是"落子前"的视角
                ch_state = pretrain_env.get_channel_state(board, current_player)
                policy = next_moves.copy()
                
                # 记录样本
                all_samples.append((ch_state, policy, value))
                valid_count += 1
                
                # 推演环境到下一步 (这会改变 env 内部的棋盘状态)
                pretrain_env.step(next_move)

            # 划分训练集和验证集（按局划分，避免同一局的帧同时出现在两个集合里）
            split = int(len(all_samples) * (1 - val_ratio))
            train_samples = all_samples[:split]
            val_samples   = all_samples[split:]

            for s in train_samples:
                memory.append(s)

            print(f"有效样本数: {valid_count}, 训练集: {len(train_samples)}, 验证集: {len(val_samples)}")

            # 预先把验证集转成tensor，避免每次重复转换
            val_states  = torch.FloatTensor(np.array([s[0] for s in val_samples])).to(device)
            val_actions = torch.FloatTensor(np.array([s[1] for s in val_samples])).to(device)
            val_values  = torch.FloatTensor(np.array([s[2] for s in val_samples])).to(device).view(-1, 1)

            pretrain_optimizer = torch.optim.AdamW(self.network.parameters(), lr=0.005, weight_decay=1e-4)
            self.network.train()

            best_val_loss = float('inf')
            patience = 0
            max_patience = 100  # 连续10次val_loss不下降则停止

            for step in range(steps):
                transitions = memory.sample(batch_size)
                states, actions, values = zip(*transitions)

                states  = np.array(states)
                actions = np.array(actions)
                values  = np.array(values)

                #print(np.unique(values,return_counts=True))

                # 旋转增强
                aug_states, aug_actions, aug_values = [], [], []
                for k in range(4):
                    rotated_states  = np.rot90(states,  k=k, axes=(2, 3)).copy()
                    rotated_actions = np.rot90(
                        actions.reshape(-1, self.board_size, self.board_size),
                        k=k, axes=(1, 2)
                    ).copy().reshape(-1, self.board_size * self.board_size)
                    aug_states.append(rotated_states)
                    aug_actions.append(rotated_actions)
                    aug_values.append(values)

                aug_states  = np.concatenate(aug_states,  axis=0)
                aug_actions = np.concatenate(aug_actions, axis=0)
                aug_values  = np.concatenate(aug_values,  axis=0)

                batch_states  = torch.FloatTensor(aug_states).to(device)
                batch_actions = torch.FloatTensor(aug_actions).to(device)
                batch_values  = torch.FloatTensor(aug_values).to(device).view(-1, 1)

                p_logits, v = self.network(batch_states)

                value_loss = F.mse_loss(v, batch_values) 

                occupied_mask = (batch_states[:, 0:1, :, :] + batch_states[:, 1:2, :, :] > 0).float()
                occupied_mask = occupied_mask.view(-1, self.board_size * self.board_size) 

                log_p = F.log_softmax(p_logits, dim=1)
                policy_loss_ce = -torch.mean(torch.sum(batch_actions * log_p, dim=1))
                # illegal_penalty = (
                #     p_logits * occupied_mask
                # ).pow(2).mean()
                policy_loss = policy_loss_ce
                pred_probs = F.softmax(p_logits, dim=1)
                illegal_prob_sum = torch.sum(pred_probs * occupied_mask, dim=1).mean()
                total_loss = 5 * value_loss + policy_loss_ce  + 10 * illegal_prob_sum

                pretrain_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                pretrain_optimizer.step()

                if step % 20 == 0:
                    current_time = datetime.now()
                    self.network.eval()
                    with torch.no_grad():
            # network policy probs
                        pred_probs = F.softmax(p_logits, dim=1)
                        # 2. 计算 Pred (网络预测) 的动作数量和 Top-5
                        pred_nonzero_counts = (pred_probs > 1e-6).sum(dim=1).float().mean()
                        pred_top5_vals, _ = torch.topk(pred_probs, k=5, dim=1)
                        pred_top5_mean = pred_top5_vals.mean(dim=0)
                        illegal_prob_sum = torch.sum(pred_probs * occupied_mask, dim=1).mean()
                            # 1. 计算 Top-K 击中率 (比如 Top-5)
                        # 如果 Minimax 的最佳点落在网络预测的 Top-5 里，就算对
                        top_k = 5
                        _, pred_topk_indices = torch.topk(pred_probs, k=top_k, dim=1) # (B, 5)
                        target_indices = batch_actions.argmax(dim=1).unsqueeze(1)      # (B, 1)
                        
                        # 判断 target 是否在 pred_topk 中
                        hit = (pred_topk_indices == target_indices).any(dim=1).float().mean()
                        
                        # 2. 计算概率覆盖度 (网络给目标点分配了多少概率)
                        # 这是一个比 Hard Acc 更温和的指标
                        target_probs = torch.gather(pred_probs, 1, target_indices).mean()
                        # ==================================================

                    # 修改打印格式
                    print(
                        #f"p_loss={policy_loss_ce.item():.4f} | "
                        # f"v_loss={value_loss.item():.4f} | "
                        #f"pred_H={pred_entropy.item():.4f} | "
                        #f"tar_H={target_entropy.item():.4f} | "
                        f"illegal={illegal_prob_sum.item():.4f} | "
                        #f"illegal_penalty={illegal_penalty.item():.4f} | "
                        f"Top5_Hit={hit.item():.3f} | "       # 新指标
                        f"Tar_Prob={target_probs.item():.3f} | "  # 新指标
                        # 打印动作数量
                        f"pred_cnt={pred_nonzero_counts.item():.1f} | "
                        #f"tar_cnt={target_nonzero_counts.item():.1f} | "
                        
                        # 打印 Top-5 概率 (格式化更易读)
                        f"pred_top5={[f'{v:.3f}' for v in pred_top5_mean.cpu().numpy()]} | "
                        #f"tar_top5={[f'{v:.3f}' for v in target_top5_mean.cpu().numpy()]}"
                    )
                    
                    
                    with torch.no_grad():
                        pred_entropy = -(
                            F.softmax(p_logits, dim=1) * F.log_softmax(p_logits, dim=1)
                        ).sum(dim=1).mean()
                        #acc = (p_logits.argmax(dim=1) == batch_actions.argmax(dim=1)).float().mean()

                        # 验证集loss，分批避免OOM
                        val_batch_size = 64
                        val_policy_losses = []
                        val_value_losses  = []
                        val_illegal_penalties = []
                        for start in range(0, len(val_samples), val_batch_size):
                            vs = val_states[start:start+val_batch_size]
                            va = val_actions[start:start+val_batch_size]
                            vv = val_values[start:start+val_batch_size]
                            
                            vp_logits, vv_pred = self.network(vs)
                            
                            vs_occupied = (vs[:, 0:1, :, :] + vs[:, 1:2, :, :] > 0).float()
                            vs_occupied = vs_occupied.view(-1, self.board_size * self.board_size)
                            
                            log_vp = F.log_softmax(vp_logits, dim=1)
                            
                            if va.dim() == 1:
                                va = F.one_hot(va, num_classes=self.board_size * self.board_size).float()
                            
                            val_policy_losses.append(-torch.mean(torch.sum(va * log_vp, dim=1)).item())
                            val_value_losses.append(F.mse_loss(vv_pred, vv).item())
                            val_illegal_penalties.append((vp_logits * vs_occupied).pow(2).mean().item())
                            
                        val_policy_loss = np.mean(val_policy_losses)
                        val_value_loss  = np.mean(val_value_losses)
                        val_illegal_penalty = np.mean(val_illegal_penalties)


                    self.network.train()

                    print(f"pretrain step {step:4d} | "
                        f"train_policy {policy_loss.item():.4f} | train_value {value_loss.item():.4f} | "
                        f"val_policy {val_policy_loss:.4f} | val_value {val_value_loss:.4f} | "
                        f"val_illegal_penalty {val_illegal_penalty:.4f} | "
                        f"pred_entropy {pred_entropy.item():.4f} | illegal_prob_sum {illegal_prob_sum.item():.4f}")

                    with open(self.agent.LOG_FILE, 'a') as f:
                        f.write(f"current_time: {current_time}, total_time: {current_time-start_time}, "
                                f"step {step}, train_policy {policy_loss.item():.4f}, "
                                #f"illegal_penalty {illegal_penalty.item():.4f}, "
                                f"val_policy {val_policy_loss:.4f}, val_value {val_value_loss:.4f}, "
                                f"val_illegal_penalty {val_illegal_penalty:.4f}\n")

                    # early stopping：以val_policy_loss为准
                    if val_policy_loss < best_val_loss - 0.001:
                        best_val_loss = val_policy_loss
                        patience = 0
                        torch.save(self.network.state_dict(), self.agent.MODEL_FILE_RESTORE)
                        print(f"  ✓ 新最优val_loss={best_val_loss:.4f}，已保存模型")
                    else:
                        patience += 1
                        if patience >= max_patience:
                            print(f"Early stopping at step {step}，val_loss连续{max_patience}次未改善")
                            break

            # 加载最优模型
            self.network.load_state_dict(torch.load(self.agent.MODEL_FILE_RESTORE, map_location=device))
            print("已加载最优预训练模型")
            print("预训练完成，清空memory开始自对弈")
            memory.memory.clear()

    def build_channel_state_from_board(self,env, board: np.ndarray, current_player: int, next_move: int) -> np.ndarray:
        """
        重放棋盘到env，生成8通道输入。
        
        约束：倒数第1,3,5步（current_player的最近3步）必须是四连中的棋子。
        这样历史通道里能看到威胁逐步形成的过程。
        
        落子顺序（从早到晚）：
        ... → cur_old → opp_3 → cur_3 → opp_2 → cur_2 → opp_1 → cur_1
                                    ↑倒5        ↑倒3        ↑倒1
        倒1/3/5 = current_player的第1/2/3新落子 = 四连中的子
        """
        env.reset()
        
        # 四连位置：board上current_player的棋子，找到包含next_move的那条四连
        five_cells = self._find_four_cells(board, current_player, next_move)
        
        # 从四连中随机选3颗作为倒数1/3/5步，剩1颗归入"早期"
        np.random.shuffle(five_cells)
        late_cur  = five_cells[:3]   # 倒数第1,3,5步（最近3步）
        early_cur = five_cells[3:]   # 剩余1颗，归入早期落子
        
        # current_player 的其余棋子（非四连）
        all_cur = np.argwhere(board == current_player).tolist()
        four_set = set(map(tuple, five_cells.tolist()))
        other_cur = [p for p in all_cur if tuple(p) not in four_set]
        np.random.shuffle(other_cur)
        
        # 对手棋子随机打乱
        opp_pos = np.argwhere(board == -current_player).tolist()
        np.random.shuffle(opp_pos)
        
        # 切出对手的"最近3步"和"早期"
        # 对手步数 = current_player步数 或少1步
        n_cur_total = len(all_cur)          # current已落子数（不含next_move）
        n_opp_total = len(opp_pos)
        
        # 对手最近3步（穿插在cur的最近3步之间）
        late_opp  = opp_pos[:3]
        early_opp = opp_pos[3:]
        
        # 构造早期落子序列（黑白交替，顺序随意）
        early_all = []
        ec = list(early_cur) + list(other_cur)
        eo = list(early_opp)
        np.random.shuffle(ec)
        np.random.shuffle(eo)
        
        # 黑先手，交替填入
        # current_player=1(黑)先走，current_player=-1(白)后走
        if current_player == 1:
            # 黑先：早期序列 黑白交替
            for i in range(max(len(ec), len(eo))):
                if i < len(ec): early_all.append((current_player,  ec[i]))
                if i < len(eo): early_all.append((-current_player, eo[i]))
        else:
            # 白后：早期序列 黑白交替（黑先）
            for i in range(max(len(ec), len(eo))):
                if i < len(eo): early_all.append((-current_player, eo[i]))
                if i < len(ec): early_all.append((current_player,  ec[i]))
        
        # 构造最近6步（倒5~倒1），穿插 cur 和 opp
        # 顺序：cur_late[2], opp_late[2], cur_late[1], opp_late[1], cur_late[0], opp_late[0]
        #        ↑倒数第5步                  ↑倒数第3步                  ↑倒数第1步
        recent = []
        for i in range(2, -1, -1):
            if i < len(late_cur):
                recent.append((current_player,  late_cur[i]))
            if i < len(late_opp):
                recent.append((-current_player, late_opp[i]))
        
        # 完整落子序列
        full_sequence = early_all + recent
        
        # 重放到env
        for player_turn, pos in full_sequence:
            r, c = pos[0], pos[1]
            # 强制env当前player和落子方一致（env自动交替，需要校正）
            if env.current_player != player_turn:
                # 顺序有偏差时跳过（理论上不应发生，加保护）
                continue
            action = int(r) * env.board_size + int(c)
            env.step(action)
        
        return env.get_channel_state(board, current_player)


    def _find_four_cells(self,board: np.ndarray, player: int, next_move: int) -> np.ndarray:
        """
        找到包含 next_move 落点的那条四连的4颗棋子坐标。
        next_move 是空格，四连是board上已有的4颗 player 的子。
        返回 shape (4, 2) 的坐标数组。
        """
        SIZE = board.shape[0]
        DIRECTIONS = [(0,1),(1,0),(1,1),(1,-1)]
        nr_target, nc_target = next_move // SIZE, next_move % SIZE
        
        flat = board.ravel()
        
        # 枚举所有5格窗口，找包含next_move且有4颗player子的窗口
        for dr, dc in DIRECTIONS:
            for r in range(SIZE):
                for c in range(SIZE):
                    er, ec = r + dr*4, c + dc*4
                    if not (0 <= er < SIZE and 0 <= ec < SIZE):
                        continue
                    cells = [(r + dr*i, c + dc*i) for i in range(5)]
                    # 窗口必须包含 next_move 位置
                    if (nr_target, nc_target) not in cells:
                        continue
                    vals = [board[cr, cc] for cr, cc in cells]
                    if vals.count(player) == 4 and vals.count(0) == 1:
                        # 返回4颗player的棋子坐标
                        four = np.array([list(cells[i]) for i in range(5) if vals[i] == player])
                        return four  # shape (4, 2)
        
        # 找不到时兜底：返回board上所有player的子
        return np.argwhere(board == player)
