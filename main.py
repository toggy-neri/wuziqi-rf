from datetime import datetime
import argparse
import time
import yaml
import os
import pickle
import json
import random
import builtins

#chess GUI
from pic import PolicyVisualizer
from wuziqi_gui import WuziqiGUI,AiMatch

#chess control
from wuziqi_env import WuziqiEnv

#pretrain module
from pretrain import pretrainer


from minimax import Minimax
from mcts import MCTS, TreeNode
from network import Network
from experience_replay import ReplayMemory
from torch.nn import functional as F
import torch

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class Agent:
    def __init__(self,hyperparameters_set,is_training=True):
        with open(f'hyperparameters.yaml', 'r') as file:
            all_hyperparameters = yaml.safe_load(file)
            hyperparameters = all_hyperparameters[hyperparameters_set]
        
        self.board_size = hyperparameters['board_size']
        self.residual_blocks = hyperparameters['residual_blocks']
        self.exploration_factor = hyperparameters['exploration_factor']
        self.is_self_play = hyperparameters['is_self_play']
        self.self_play_num = hyperparameters['self_play_num']
        self.run_dir = hyperparameters['run_dir']
        self.restore_dir = hyperparameters['restore_dir']
        self.restore_epoch = hyperparameters['restore_epoch']
        self.is_training = is_training
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.inference_batch_size = hyperparameters['inference_batch_size']
        self.optimizer_batch_size = hyperparameters['optimizer_batch_size']
        self.update_freq = hyperparameters['update_freq']
        self.search_num = hyperparameters['search_num']
        self.warmup_steps = hyperparameters['warmup_steps']
        self.is_muti_optimizer = hyperparameters['is_muti_optimizer']
        self.num_optimizer = hyperparameters['num_optimizer']
        self.lr = hyperparameters['lr']
        self.debug_ai_minimax_board = hyperparameters.get('debug_ai_minimax_board', False)
        self.debug_training_print = hyperparameters.get('debug_training_print', False)
        self.use_virtual_loss = hyperparameters.get('use_virtual_loss', False)
        self.virtual_loss = hyperparameters.get('virtual_loss', 1.0)

        #pretrain
        self.pretrainer = None
        self.is_pretrain = hyperparameters['is_pretrain']            
        self.pretrain_steps = hyperparameters['pretrain_steps']
        self.pretrain_file = hyperparameters['pretrain_file']

        self.is_continue_training = hyperparameters['is_continue_training']
        #store the model file and info 
        self.exist_model_name = hyperparameters['exist_model_name']
        self.MODEL_FILE = os.path.join(self.run_dir, f'{self.exist_model_name}.pt')
        
        if not self.is_pretrain:
            self.MODEL_FILE_RESTORE = os.path.join(self.restore_dir, f'{hyperparameters_set}.pt')
        else:
            self.MODEL_FILE_RESTORE = os.path.join(self.restore_dir, f'{hyperparameters_set}_pretrain.pt')
        self.BEST_MODEL_FILE_RESTORE = os.path.join(self.restore_dir, f'{hyperparameters_set}_best.pt')
        self.LOG_FILE = os.path.join(self.restore_dir, f'{hyperparameters_set}.txt')
        self.LOG_FILE_OPTIMIZE = os.path.join(self.restore_dir, f'{hyperparameters_set}_optimize.txt')  
        self.TRAINING_DATA_LOG = os.path.join(self.restore_dir, f'{hyperparameters_set}_training_data.log')

        self.MEMORY_FILE = os.path.join(self.restore_dir, f'{self.exist_model_name}_memory.pkl') 
        os.makedirs(self.restore_dir, exist_ok=True)

    def _reset_training_data_log_if_needed(self, training_index):
        if (training_index - 1) % 5 != 0:
            return

        with open(self.TRAINING_DATA_LOG, 'w', encoding='utf-8') as f:
            f.write(f"# training data log reset at training {training_index}, time={datetime.now().strftime(DATE_FORMAT)}\n")

    def _build_training_step_log(self, step, player, action, board_before, policy, reward, done, info, board_after):
        if action is None:
            position = None
            action_value = None
        else:
            row, col = divmod(int(action), self.board_size)
            position = [int(row), int(col)]
            action_value = int(action)

        return {
            "step": int(step),
            "player": int(player),
            "action": action_value,
            "position": position,
            "board_before": board_before.tolist(),
            "board_after": board_after.tolist(),
            "policy": policy.tolist(),
            "reward": float(reward),
            "done": bool(done),
            "info": info,
        }

    def _json_default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return str(obj)

    def _write_training_data_log(self, training_index, winner, steps, epoch_time):
        record = {
            "training": int(training_index),
            "time": datetime.now().strftime(DATE_FORMAT),
            "winner": int(winner) if winner is not None else None,
            "epoch_time": float(epoch_time),
            "steps": steps,
        }
        with open(self.TRAINING_DATA_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=self._json_default) + "\n")

    def _debug_print_ai_minimax_board(self, game_index, step, actor, player, action, env):
        side = "Black" if player == 1 else "White"
        if action is None:
            position = "pass"
        else:
            row, col = divmod(int(action), self.board_size)
            position = f"({row}, {col})"

        print(
            f"\n[AI-vs-Minimax Debug] game={game_index} step={step} "
            f"actor={actor} side={side} action={action} pos={position}"
        )
        print(env.render())

    def _advance_or_rebuild_root(self, root, action):
        if root is not None and action in root.children:
            root = root.children[action]
            root.parent = None
            return root
        return TreeNode(parent=None)

    def run(self,is_training=True,render=False):
        #create environment
        start_time = datetime.now()
        env = WuziqiEnv(board_size=self.board_size)
        
        self.network = Network(board_size=self.board_size).to(device)   

        
        if not self.is_continue_training and is_training:
             pass
        else:
            try:
                self.network.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            except:
                raise FileNotFoundError(f"Model file {self.MODEL_FILE} not found")  
        
        if self.is_pretrain and is_training:
            self.pretrainer = pretrainer(self)
            memory = ReplayMemory(500000)
            self.pretrainer.pretrain(self.pretrain_file, memory, self.pretrain_steps, self.optimizer_batch_size)
            raise SystemExit(0)


        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            resore_count = 0
            #store path of the mddel
            os.makedirs(self.run_dir, exist_ok=True)
            if os.path.exists(self.MEMORY_FILE):
                with open(self.MEMORY_FILE, 'rb') as f:
                    memory = pickle.load(f)
                if self.debug_training_print:
                    print(f"Loaded memory with {len(memory)} samples")
            else:
                if self.debug_training_print:
                    print("No memory file found, starting fresh")
        else:
            ai_match = AiMatch(
                env,
                self.network,
                self.search_num,
                self.inference_batch_size,
                use_virtual_loss=self.use_virtual_loss,
                virtual_loss=self.virtual_loss,
            )
            ai_match.run()

        #create network
        if not self.is_self_play:
            minimax = Minimax(self.board_size)
        
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(self.network.conv_input.parameters())
                    + list(self.network.bn_input.parameters())
                    + list(self.network.residual_blocks.parameters()),
                    "lr": self.lr,
                },
                {
                    "params": list(self.network.policy_head.parameters())
                    + list(self.network.bn_policy.parameters())
                    + list(self.network.policy_fc.parameters()),
                    "lr": self.lr,
                },
                {
                    "params": list(self.network.value_head.parameters())
                    + list(self.network.bn_value.parameters())
                    + list(self.network.value_fc1.parameters())
                    + list(self.network.value_fc2.parameters()),
                    "lr": self.lr,
                },
            ],
            weight_decay=1e-4,
        )


        #create mcts
        mcts = MCTS(
            self.network,
            env,
            is_training=self.is_training,
            use_virtual_loss=self.use_virtual_loss,
            virtual_loss=self.virtual_loss,
        )

        #update rate   
        sync_count = 0
        restore_count = 0
        policy_loss = float('inf')
        value_loss = float('inf')
        memory_extract_var = 7
        low_policy_loss_streak = 0
        epochs_since_var_up = 0

        total_games = 0
        for i in range(self.self_play_num):
            training_index = i + 1
            if is_training:
                self._reset_training_data_log_if_needed(training_index)
            time_start = time.time()

            root = TreeNode(parent=None)

            game_history = []
            state = env.reset()
            done = False
            current_step = 0
            training_steps_log = []
            if not self.is_self_play:
                # 每局开始时随机分配
                ai_player = random.choice([1, -1])
                minimax_player = -ai_player
                if self.debug_training_print:
                    print(f"本局 AI执{'黑' if ai_player == 1 else '白'}，Minimax执{'黑' if minimax_player == 1 else '白'}")
            while not done:
                current_step += 1
                current_time = datetime.now()
                if self.is_self_play or not self.is_training:
                    winning_moves = mcts.get_winning_moves(env.board, env.current_player)
                    defend_moves  = mcts.get_winning_moves(env.board, -env.current_player)
                    if len(winning_moves) > 0:
                        # 己方有必杀点，直接赢
                        forced_moves = winning_moves
                    elif len(defend_moves) > 0:
                        # 对手有必杀点，必须封堵
                        forced_moves = defend_moves
                    else:
                        forced_moves = np.array([])

                    if len(forced_moves) > 0:
                        policy = np.zeros(self.board_size ** 2, dtype=np.float32)
                        policy[forced_moves] = 1.0 / len(forced_moves)
                        action = int(forced_moves[0])
                        root = TreeNode(parent=None)
                    else:
                        if root is None:
                            if self.debug_training_print:
                                print("[Train] root is None before MCTS search, rebuild root")
                            root = TreeNode(parent=None)
                        mcts.search(root,self.inference_batch_size,self.search_num,self.exploration_factor)
                        policy = mcts.get_policy(root,self.board_size)
                        #choose the best move 
                        action,root = mcts.choose(root,is_training,current_step=current_step)
                        if policy.sum() <= 0 and action is not None:
                            policy[int(action)] = 1.0
                        if root is not None:
                            root.parent = None
                    board_before = env.board.copy()
                    step_player = env.current_player

                    game_history.append((state.copy(),policy.copy(),env.current_player,env.last_move))
                    state, reward, done, info = env.step(action)  
                    
                    if is_training:
                        training_steps_log.append(
                            self._build_training_step_log(
                                current_step, step_player, action, board_before, policy,
                                reward, done, info, state.copy()
                            )
                        )
                else:
                    if env.current_player == minimax_player:   # 黑方 = minimax
                        action, policy_matrix = minimax.get_minimax_action(
                            board          = env.board.copy(),          # 直接传棋盘
                            current_player = env.current_player,
                            depth          = 1
                        )
                        policy = policy_matrix.flatten().astype(np.float32)
                        root = self._advance_or_rebuild_root(root, action)
                          # minimax走法当做确定性policy，避免后面undefined
                    else:                         # 白方 = 你的AI (mcts)
                        if root is None:
                            if self.debug_training_print:
                                print("[Train] root is None before MCTS search, rebuild root")
                            root = TreeNode(parent=None)
                        mcts.search(root, self.inference_batch_size, self.search_num, self.exploration_factor)
                        policy = mcts.get_policy(root, self.board_size)
                        action, root = mcts.choose(root, is_training,current_step=current_step)
                        if policy.sum() <= 0 and action is not None:
                            policy[int(action)] = 1.0
                    board_before = env.board.copy()
                    step_player = env.current_player
                    if env.current_player == minimax_player :
                        game_history.append((state.copy(),policy.copy(),-env.current_player,env.last_move))
                    state, reward, done, info = env.step(action)  
                    if self.debug_ai_minimax_board:
                        actor = "Minimax" if step_player == minimax_player else "AI"
                        self._debug_print_ai_minimax_board(
                            training_index,
                            current_step,
                            actor,
                            step_player,
                            action,
                            env
                        )
                    if is_training:
                        training_steps_log.append(
                            self._build_training_step_log(
                                current_step, step_player, action, board_before, policy,
                                reward, done, info, state.copy()
                            )
                        )

                
                if done:
                    winner = info["winner"]
                    if self.debug_training_print:
                        print(f"[Game End] Winner: {winner}")
                    total_games += 1
        
                        
                    tail_count = min(len(game_history), memory_extract_var)
                    selected_history = game_history[-tail_count:] if tail_count > 0 else []
                    for s,p,player,last_move in selected_history:  #注意不要保证重名
                        if winner == 0:
                            value = 0
                        else:
                            value = 1 if player == winner else -1
                        ch_state = env.get_channel_state(s,player)
                        memory.append((ch_state, p,value))
                    if is_training:
                        self._write_training_data_log(training_index, winner, training_steps_log, time.time() - time_start)
                    game_history = []
                    break

            sync_count+=1
            restore_count+=1
            current_time = datetime.now()
            time_end = time.time()
            epochs_since_var_up += 1

            if sync_count % self.update_freq == 0:
                

                if len(memory) < self.warmup_steps:
                    if self.debug_training_print:
                        print(f"Memory too small ({len(memory)}), skipping training")
                else:
                    value_loss,policy_loss = self.optimize(memory, self.optimizer_batch_size)
                    if policy_loss < 2.0 and value_loss < 0.5:
                        low_policy_loss_streak += 1
                        if low_policy_loss_streak >= 10:
                            old_var = memory_extract_var
                            memory_extract_var += 1
                            memory.memory.clear()
                            epochs_since_var_up = 0
                            if self.debug_training_print:
                                print(f"memory_extract_var up: {old_var} -> {memory_extract_var}, streak reset, replay memory cleared")
                            with open(self.LOG_FILE, 'a') as f:
                                f.write(f"var up.current_time: {current_time},total_time: {current_time-start_time}, Epoch {restore_count},epoch_time: {time_end-time_start:.3f}s, old_extract_var: {old_var}, now_extract_var: {memory_extract_var}, policy_loss: {policy_loss:.4f}, value_loss: {value_loss:.4f}, memory_cleared: True\n")
                            low_policy_loss_streak = 0
                    else:
                        low_policy_loss_streak = 0

                if self.debug_training_print:
                    print(f"current_time: {current_time},total_time: {current_time-start_time}, Epoch {restore_count},epoch_time: {time_end-time_start:.3f}s, Policy Loss {policy_loss:.4f}, Value Loss {value_loss:.4f}")

            if epochs_since_var_up >= 500:
                old_var = memory_extract_var
                memory_extract_var = max(1, memory_extract_var - 1)
                low_policy_loss_streak = 0
                epochs_since_var_up = 0
                msg = (
                    f"memory_extract_var down: {old_var} -> {memory_extract_var}; "
                    f"no var up for 200 epochs, replay memory kept"
                )
                if self.debug_training_print:
                    print(msg)
                with open(self.LOG_FILE, 'a') as f:
                    f.write(f"var down.current_time: {current_time},total_time: {current_time-start_time}, Epoch {restore_count},epoch_time: {time_end-time_start:.3f}s, old_extract_var: {old_var}, now_extract_var: {memory_extract_var}, memory_cleared: False\n")

            if restore_count % self.restore_epoch == 0:

                with open(self.LOG_FILE, 'a') as f:
                    f.write(f"current_time: {current_time},total_time: {current_time-start_time}, Epoch {restore_count},epoch_time: {time_end-time_start:.3f}s, Policy Loss {policy_loss:.4f}, Value Loss {value_loss:.4f}\n")
                    torch.save(self.network.state_dict(), self.MODEL_FILE_RESTORE)
                

                with open(self.MEMORY_FILE, 'wb') as f:
                    pickle.dump(memory, f)

                   #optimize the network

    #optimize the network
    def optimize(self, memory, batch_size=64):
        print = builtins.print if self.debug_training_print else (lambda *args, **kwargs: None)
        self.network.train()
        for _ in range(self.num_optimizer if self.is_muti_optimizer else 1):
            transitions = memory.sample(batch_size)
            states, actions, values = zip(*transitions)
            device = next(self.network.parameters()).device
            
            


            states = np.array(states)    # (B, 4, 15, 15)
            actions = np.array(actions)  # (B, 225)
            values = np.array(values)    # (B,)

            # 旋转增强：对每个样本生成4个旋转版本
            aug_states = []
            aug_actions = []
            aug_values = []

            for k in range(4):  # k=0,1,2,3 对应 0°,90°,180°,270°
                # 旋转棋盘：state的后两个维度是棋盘(H,W)，对axis=(2,3)旋转
                rotated_states = np.rot90(states, k=k, axes=(2, 3)).copy()

                rotated_actions = actions.reshape(-1, self.board_size, self.board_size)
                rotated_actions = np.rot90(rotated_actions, k=k, axes=(1, 2)).copy()
                rotated_actions = rotated_actions.reshape(-1, self.board_size * self.board_size)

                aug_states.append(rotated_states)
                aug_actions.append(rotated_actions)
                aug_values.append(values)

            # 拼接成 4*B 的大batch
            aug_states  = np.concatenate(aug_states,  axis=0)
            aug_actions = np.concatenate(aug_actions, axis=0)
            aug_values  = np.concatenate(aug_values,  axis=0)

            batch_states  = torch.FloatTensor(aug_states).to(device)
            batch_actions = torch.FloatTensor(aug_actions).to(device)
            batch_values  = torch.FloatTensor(aug_values).to(device).view(-1, 1)
 
            p_logits, v = self.network(batch_states)
            if self.debug_training_print:
                print(f"v mean: {v.mean().item():.3f}, std: {v.std().item():.3f}")
            print(f"V 预测值样例: {v[:5].detach().cpu().numpy().flatten()}") # 打印前5个预测值
            print(f"V 真实值样例: {batch_values[:5].cpu().numpy().flatten()}")
            value_loss = F.mse_loss(v, batch_values) 

            # 2. 构建绝对非法位置的 Mask (已有棋子的地方)
            # 假设 batch_states 的前两个通道是当前方和对手方
            # shape: (B, C, H, W)
            occupied_mask = (batch_states[:, 0:1, :, :] + batch_states[:, 1:2, :, :] > 0).float()
            occupied_mask = occupied_mask.view(-1, self.board_size * self.board_size) # (B, 225)



            
            log_p = F.log_softmax(p_logits, dim=1)
            
            
            policy_loss_ce = -torch.mean(
                torch.sum(batch_actions * log_p, dim=1)
            )
            pred_probs = F.softmax(p_logits, dim=1)
            illegal_prob_sum = torch.sum(pred_probs * occupied_mask, dim=1).mean()
            policy_loss = policy_loss_ce
            total_loss = 5 * value_loss + policy_loss_ce + 1000*illegal_prob_sum








            vis = PolicyVisualizer(save_dir='./runs/policy_logs', cmap='viridis')

            # ... 训练循环中 ...
            with torch.no_grad():
                
                
                
                
                
                
                
                

                # 1. 画单个样本的预测分布 (比如画 Batch 里的第 0 个样本)
                sample_idx = 0
                
                # Prepare board channels and policy distributions for one summary image.
                board_vis = batch_states[sample_idx, 0:2, :, :].cpu().numpy()
                
                # 2. 准备概率数据 (需要 Softmax 才是概率)
                pred_probs = F.softmax(p_logits[sample_idx], dim=0).cpu().numpy().reshape(self.board_size, self.board_size)
                
                target_board = batch_actions[sample_idx].reshape(self.board_size, self.board_size).cpu().numpy()

                # 3. 计算上一步位置
                # 取第6个通道
                last_move_tensor = batch_states[sample_idx, 6, :, :] 
                
                # 找到非零的坐标 (假设只有一个点为1)
                last_coords = (last_move_tensor > 0).nonzero()
                
                last = None
                if last_coords.nelement() > 0:
                    coord = last_coords[0] 
                    
                    # ✅ coord 的第0个元素是行，第1个元素是列
                    r = coord[0].item()
                    c = coord[1].item()
                    last = (r, c)
                else:
                    print(f"Warning: No last move found in channel 6 for sample {sample_idx}")

                # 2. 生成对比图 (推荐)
                vis.save_policy_summary(
                    pred_data=pred_probs,
                    target_data=target_board,
                    board_data=board_vis,
                    probs_data=pred_probs,
                    table_name="Step",
                    last_move=last,
                    epoch=0
                )

                # 生成纯净棋盘（查看当前局面）
            # 修改打印格式
            print(
                f"p_loss={policy_loss_ce.item():.4f} | "
                f"v_loss={value_loss.item():.4f} | "
                f"illegal_penalty={illegal_prob_sum.item():.4f} | "
                
                # # 打印动作数量
                
                # # 打印 Top-5 概率 (格式化更易读)
            )
            self.optimizer.zero_grad()
            total_loss.backward()
            #cursor 
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
        return value_loss.item(),policy_loss.item() 
    
    def overfit_test(self, memory, steps=2000,batch_size=128):

        
        """过拟合测试：用固定batch反复训练，验证网络能否收敛"""
        print("\n========== 过拟合测试开始 ==========")
        
        all_transitions = list(memory.memory)  # 取所有数据
        pos = [t for t in all_transitions if t[2] == 1]
        neg = [t for t in all_transitions if t[2] == -1]
        
        print(f"正样本: {len(pos)}, 负样本: {len(neg)}")
        
        half = batch_size // 2
        balanced =  memory.sample(batch_size)
        # 在 overfit_test 开始时加
        print("Target policy 非零统计:")
        for i, (s, a, v) in enumerate(balanced[:5]):
            nonzero = np.count_nonzero(a)
            top3 = np.argsort(a)[-3:][::-1]
            print(f"样本{i}: 非零动作数={nonzero}, top3值={a[top3]}, top3位置={top3}")
        states, actions, values = zip(*balanced)
            
        device = next(self.network.parameters()).device
        batch_states  = torch.FloatTensor(np.array(states)).to(device)
        batch_actions = torch.FloatTensor(np.array(actions)).to(device)
        batch_values  = torch.FloatTensor(np.array(values)).to(device).view(-1, 1)
        print(f"batch_actions sum per sample: {batch_actions.sum(dim=1)[:5]}")
        print(f"batch_actions max per sample: {batch_actions.max(dim=1).values[:5]}")
        print(f"batch_actions nonzero per sample: {(batch_actions > 0).sum(dim=1)[:5]}")
        
        print(f"States unique count: {len(np.unique(states, axis=0))}")
        test_optimizer = torch.optim.Adam(self.network.parameters(), lr=0.005)
        for step in range(steps):
            p_logits, v = self.network(batch_states)
            v = v.view(-1, 1)
            value_loss  = F.mse_loss(v, batch_values)
            log_p       = F.log_softmax(p_logits, dim=1)
            policy_loss = -torch.mean(torch.sum(batch_actions * log_p, dim=1))
            loss        = value_loss + policy_loss
            
            test_optimizer.zero_grad()
            loss.backward()
            
            test_optimizer.step()
            if step % 100 == 0:
                print(f"step {step:4d} | value_loss: {value_loss.item():.4f} | "
                    f"policy_loss: {policy_loss.item():.4f} | "
                    f"v_std: {v.std().item():.4f}")
                pred_move = torch.argmax(p_logits, dim=1)
                target_move = torch.argmax(batch_actions, dim=1)
                accuracy = (pred_move == target_move).float().mean()
                print(f"Policy Accuracy: {accuracy:.2%}")

                p_dist = F.softmax(p_logits[0], dim=0).detach().cpu().numpy()
                t_dist = batch_actions[0].cpu().numpy()
                print(f"Top 3 Predicted moves: {np.argsort(p_dist)[-3:][::-1]}")
                print(f"Top 3 Target moves: {np.argsort(t_dist)[-3:][::-1]}")
                            
        
        print("========== 过拟合测试结束 ==========\n")
    def set_tower_grad(self, requires_grad: bool):
        for param in self.network.conv_input.parameters():
            param.requires_grad = requires_grad
        for param in self.network.bn_input.parameters():
            param.requires_grad = requires_grad
        for param in self.network.residual_blocks.parameters():
            param.requires_grad = requires_grad
        state = "解冻" if requires_grad else "冻结"
        print(f"残差塔已{state}")
  # 去重后的必杀点
def play_gui():
    game = WuziqiGUI(board_size=15, cell_size=40)
    game.run()

def test_env():
    env = WuziqiEnv(board_size=15)
    state = env.reset()
    
    print("初始棋盘:")
    print(env.render())
    
    moves = [(7, 7), (7, 8), (8, 7), (8, 8), (6, 7), (9, 6), (5, 7)]
    
    for i, move in enumerate(moves):
        state, reward, done, info = env.step(move)
        print(f"\n第{i+1}步: {move}")
        print(env.render())
        print(f"奖励: {reward}, 结束: {done}, 信息: {info}")
        
        if done:
            break



def main():
    parser = argparse.ArgumentParser(description='Wuziqi Game')
    parser.add_argument('hyperparameters', type=str, default='test1',
                       help='超参数配置文件')
    
    parser.add_argument('--train', action='store_true', help='是否训练')
    args = parser.parse_args()
    agent = Agent(args.hyperparameters,args.train)

    
    if args.train:
        agent.run(is_training=True,render=False)
    else:
        agent.run(is_training=False,render=True)

if __name__ == "__main__":
    main()
