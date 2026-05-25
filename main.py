from datetime import datetime
import sys
import argparse
import time
import yaml
import os
import pickle

#chess GUI
from wuziqi_gui import WuziqiGUI,AiMatch

#chess control
from wuziqi_env import WuziqiEnv

#pretrain module
from pretrain import pretrainer


from minimax import Minimax
from mcts import MCTS, TreeNode
from network import Network, Residual_block
from experience_replay import ReplayMemory
import copy 
from torch import nn
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

        self.MEMORY_FILE = os.path.join(self.restore_dir, f'{self.exist_model_name}_memory.pkl') 
        os.makedirs(self.restore_dir, exist_ok=True)
        #self.GRAPH_FILE = os.path.join(self.run_dir, f'{hyperparameters_set}.png')
        
    def run(self,is_training=True,render=False):
        #create environment
        start_time = datetime.now()
        env = WuziqiEnv(board_size=self.board_size)
        
        self.network = Network(board_size=self.board_size).to(device)   

        
        if not self.is_continue_training and is_training:
             pass#self.network = Network(board_size=self.board_size).to(device)   
        else:
            try:
                self.network.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
                #self.network = torch.load(self.MODEL_FILE).to(device)
            except:
                raise FileNotFoundError(f"Model file {self.MODEL_FILE} not found")  
        
        if self.is_pretrain and is_training:
            self.pretrainer = pretrainer(self)
            memory = ReplayMemory(500000)
            self.pretrainer.pretrain(self.pretrain_file, memory, self.pretrain_steps, self.optimizer_batch_size)
            import sys
            sys.exit(0)


        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            resore_count = 0
            #store path of the mddel
            os.makedirs(self.run_dir, exist_ok=True)
            if os.path.exists(self.MEMORY_FILE):
                with open(self.MEMORY_FILE, 'rb') as f:
                    memory = pickle.load(f)
                print(f"Loaded memory with {len(memory)} samples")
            else:
                print("No memory file found, starting fresh")
        else:
            ai_match = AiMatch(env, self.network,self.search_num,self.inference_batch_size)
            ai_match.run()

        #create network
        if not self.is_self_play:
            minimax = Minimax(self.board_size)
        
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.lr,weight_decay=1e-4)#此处更换


        #create mcts
        mcts = MCTS(self.network,env,is_training=self.is_training)

        #update rate   
        sync_count = 0
        restore_count = 0
        policy_loss = float('inf')
        value_loss = float('inf')
        best_loss = float('inf')

        ai_wins = 0
        minimax_wins = 0
        draws = 0
        total_games = 0
        #for i in range(self.self_play_num):
        for i in range(self.self_play_num):
            #time.sleep(5)
            time_start = time.time()
            root = TreeNode(parent=None)
            game_history = []
            state = env.reset()
            done = False
            current_step = 0
            #flag = 0
            if not self.is_self_play:
                import random
                # 每局开始时随机分配
                ai_player = random.choice([1, -1])
                minimax_player = -ai_player
                print(f"本局 AI执{'黑' if ai_player == 1 else '白'}，Minimax执{'黑' if minimax_player == 1 else '白'}")
            while not done:
                current_step += 1
                #flag += 1
                current_time = datetime.now()
                #print( f"Time: {current_time.strftime(DATE_FORMAT)}")
                if self.is_self_play or not self.is_training:
                    winning_moves = self.get_winning_moves(env.board, env.current_player)
                    defend_moves  = self.get_winning_moves(env.board, -env.current_player)
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
                        #t_search = time.time()
                        #search the tree,return policy(15*15)
                        mcts.search(root,self.inference_batch_size,self.search_num,self.exploration_factor)
                        #t1 = time.time()
                        #visit_board = np.zeros((self.board_size, self.board_size))
                        # for action, child in root.children.items():
                        #     row = action // self.board_size
                        #     col = action % self.board_size
                        #     visit_board[row, col] = child.visits
                        # print("\nRoot Children Visits:")
                        # print(visit_board)
                        # print(env.board)
                        policy = mcts.get_policy(root,self.board_size)
                        #choose the best move 
                        action,root = mcts.choose(root,is_training,current_step=current_step)
                        # print(f"action: {action}",f"row: {action // self.board_size},col: {action % self.board_size}")
                        #t2 = time.time()
                        root.parent = None
                    #step forward
                    game_history.append((state.copy(),policy.copy(),env.current_player,env.last_move))
                    state, reward, done, info = env.step(action)  
                else:
                    if env.current_player == minimax_player:   # 黑方 = minimax
                        action, policy_matrix = minimax.get_minimax_action(
                            board          = env.board.copy(),          # 直接传棋盘
                            current_player = env.current_player,
                            depth          = 1
                        )
                        policy = policy_matrix.flatten().astype(np.float32)
                          # minimax走法当做确定性policy，避免后面undefined
                    else:                         # 白方 = 你的AI (mcts)
                        mcts.search(root, self.inference_batch_size, self.search_num, self.exploration_factor)
                        policy = mcts.get_policy(root, self.board_size)
                        action, root = mcts.choose(root, is_training,current_step=current_step)
                    #print(env.board)
                    if env.current_player == minimax_player:
                            game_history.append((state.copy(),policy.copy(),-env.current_player,env.last_move))
                    state, reward, done, info = env.step(action)  
                    #print(env.board)
                        
                        
                        
                #store the game history
                # if is_training:
                #     now_player = env.current_player
                #     if self.is_self_play:
                #         game_history.append((state.copy(),policy.copy(),-now_player,env.last_move))
                #     else:
                #         if env.current_player == minimax_player:
                #             game_history.append((state.copy(),policy.copy(),-now_player,env.last_move))
                
                #t3 = time.time()
                
                #print(f"search: {t1-t_search:.3f}s | choose: {t2-t1:.3f}s | step+copy: {t3-t2:.3f}s | flag: {flag}")
                #print(env.board)
                #
                if done:
                    winner = info["winner"]
                    total_games += 1
        
                    # # ================= 统计胜率 =================
                    # if winner == 0:
                    #     draws += 1
                    # elif winner == minimax_player:
                    #     minimax_wins += 1
                    # else:
                    #     ai_wins += 1
                        
                    # # 打印实时胜率
                    # if total_games % 5 == 0:  # 每5局打印一次
                    #     ai_wr = ai_wins / total_games * 100
                    #     mm_wr = minimax_wins / total_games * 100
                    #     dr = draws / total_games * 100
                    #     print(f"[对局 {total_games}] 🤖 AI胜率: {ai_wr:.1f}% ({ai_wins}胜) | 🧠 Minimax胜率: {mm_wr:.1f}% ({minimax_wins}胜) | 🤝 平局: {dr:.1f}%")
                    # # ============================================
                    for s,p,player,last_move in game_history:  #注意不要保证重名
                        #print(s)
                        # if not self.is_self_play:
                        #     value = 1 if player == winner else (-1+discount)
                        # else:
                        value = 1 if player == winner else -1
                        ch_state = env.get_channel_state(s,player)
                        #print(f"Policy Max Prob: {np.max(p):.4f}, Non-zero actions: {np.count_nonzero(p)}")
                        memory.append((ch_state, p,value))
                        # if len(memory) >= self.replay_memory_size:
                        #     self.optimize(memory, self.optimizer_batch_size
                    game_history = []
                    break

            sync_count+=1
            restore_count+=1
            current_time = datetime.now()
            time_end = time.time()
            #print(f"game {i} done, time: {time_end-time_start:.3f}s")

            # if restore_count == 2:  # 第一次optimize后立即测试，后续添加到超参数 todo
            #     self.overfit_test(memory)
            #     import sys; sys.exit()
            if sync_count % self.update_freq == 0:
                

                if len(memory) < self.warmup_steps:
                    print(f"Memory too small ({len(memory)}), skipping training")
                    continue
                value_loss,policy_loss = self.optimize(memory, self.optimizer_batch_size)

                print(f"current_time: {current_time},total_time: {current_time-start_time}, Epoch {restore_count},epoch_time: {time_end-time_start:.3f}s, Policy Loss {policy_loss:.4f}, Value Loss {value_loss:.4f}")

            if policy_loss < best_loss:
                best_loss = policy_loss
                with open(self.LOG_FILE, 'a') as f:
                    f.write(f"new best.current_time: {current_time},total_time: {current_time-start_time}, Epoch {restore_count},epoch_time: {time_end-time_start:.3f}s, Policy Loss {policy_loss:.4f}, Value Loss {value_loss:.4f}\n")
                    torch.save(self.network.state_dict(), self.BEST_MODEL_FILE_RESTORE)
            if restore_count % self.restore_epoch == 0:

                with open(self.LOG_FILE, 'a') as f:
                    f.write(f"current_time: {current_time},total_time: {current_time-start_time}, Epoch {restore_count},epoch_time: {time_end-time_start:.3f}s, Policy Loss {policy_loss:.4f}, Value Loss {value_loss:.4f}\n")
                    torch.save(self.network.state_dict(), self.MODEL_FILE_RESTORE)
                

                with open(self.MEMORY_FILE, 'wb') as f:
                    pickle.dump(memory, f)

                   #optimize the network

    #optimize the network
    def optimize(self, memory, batch_size=64):
        self.network.train()
        for _ in range(self.num_optimizer if self.is_muti_optimizer else 1):
            transitions = memory.sample(batch_size)
            states, actions, values = zip(*transitions)
            device = next(self.network.parameters()).device
            # 诊断数据质量
            # values_arr = np.array(values)
            # print(f"Memory size: {len(memory)}")
            # print(f"Value distribution: +1={np.sum(values_arr==1)}, -1={np.sum(values_arr==-1)}, mean={values_arr.mean():.3f}")
            
            # nonzeros = [np.count_nonzero(a) for a in actions]
            # print(f"Policy nonzero: min={min(nonzeros)}, max={max(nonzeros)}, mean={np.mean(nonzeros):.1f}")
            
            # top_vals = [np.max(a) for a in actions]
            #print(f"Policy max prob: min={min(top_vals):.3f}, max={max(top_vals):.3f}, mean={np.mean(top_vals):.3f}")
            # for i, (s, a, v) in enumerate(transitions[:5]):
            #     nonzero = np.count_nonzero(a)
            #     top3 = np.argsort(a)[-3:][::-1]

            #     log_text = (
            #         f"样本{i}: "
            #         f"非零动作数={nonzero}, "
            #         f"top3值={a[top3]}, "
            #         f"top3位置={top3}\n"
            #     )

                # with open(self.LOG_FILE_OPTIMIZE, 'a', encoding='utf-8') as f:
                #     f.write(log_text)
            states = np.array(states)    # (B, 4, 15, 15)
            actions = np.array(actions)  # (B, 225)
            values = np.array(values)    # (B,)

            # 旋转增强：对每个样本生成4个旋转版本
            aug_states = []
            aug_actions = []
            # aug_values = []

            for k in range(4):  # k=0,1,2,3 对应 0°,90°,180°,270°
                # 旋转棋盘：state的后两个维度是棋盘(H,W)，对axis=(2,3)旋转
                rotated_states = np.rot90(states, k=k, axes=(2, 3)).copy()

                # policy也要跟着旋转：先reshape成棋盘形状，旋转，再展平
                rotated_actions = actions.reshape(-1, self.board_size, self.board_size)
                rotated_actions = np.rot90(rotated_actions, k=k, axes=(1, 2)).copy()
                rotated_actions = rotated_actions.reshape(-1, self.board_size * self.board_size)

                aug_states.append(rotated_states)
                aug_actions.append(rotated_actions)
                # aug_values.append(values)

            # 拼接成 4*B 的大batch
            aug_states  = np.concatenate(aug_states,  axis=0)
            aug_actions = np.concatenate(aug_actions, axis=0)
            # aug_values  = np.concatenate(aug_values,  axis=0)

            batch_states  = torch.FloatTensor(aug_states).to(device)
            batch_actions = torch.FloatTensor(aug_actions).to(device)
            batch_values  = torch.FloatTensor(values).to(device).view(-1, 1)
            
            # states = np.array(states)    # (B, 4, 15, 15)
            # actions = np.array(actions)  # (B, 225)
            # values = np.array(values)    # (B,)

            # # ================= 修改：随机旋转增强 =================
            # # 每次只随机选一种旋转（0°, 90°, 180°, 270°），不要拼接！
            # k = np.random.randint(0, 4) 
            
            # # 旋转棋盘
            # aug_states = np.rot90(states, k=k, axes=(2, 3)).copy()

            # # policy跟着旋转
            # aug_actions = actions.reshape(-1, self.board_size, self.board_size)
            # aug_actions = np.rot90(aug_actions, k=k, axes=(1, 2)).copy()
            # aug_actions = aug_actions.reshape(-1, self.board_size * self.board_size)

            # aug_values = values # Value 不变，因为只是视角旋转，胜负归属不变
            # # ========================================================

            # batch_states  = torch.FloatTensor(aug_states).to(device)
            # batch_actions = torch.FloatTensor(aug_actions).to(device)
            # batch_values  = torch.FloatTensor(aug_values).to(device).view(-1, 1)


            # batch_states  = torch.FloatTensor(states).to(device)
            # batch_actions = torch.FloatTensor(actions).to(device)
            # batch_values  = torch.FloatTensor(values).to(device).view(-1, 1)
            #legal_mask = (batch_actions > 0).float() 
            p_logits, v = self.network(batch_states)
            print(f"v mean: {v.mean().item():.3f}, std: {v.std().item():.3f}")
            print(f"V 预测值样例: {v[:5].detach().cpu().numpy().flatten()}") # 打印前5个预测值
            print(f"V 真实值样例: {batch_values[:5].cpu().numpy().flatten()}")
            value_loss = F.mse_loss(v[:batch_size], batch_values) 
            #value_loss = F.mse_loss(v, batch_values)

            # policy cross entropy
            # mask = (batch_actions > 0).float()
            # log_p = F.log_softmax(p_logits, dim=1)
            # policy_loss = -torch.mean(
            #     torch.sum(batch_actions * log_p * mask, dim=1)
            # )



                        # ============ 诊断：Policy 对比 ============
            # with torch.no_grad():
            #     pred_probs = F.softmax(p_logits, dim=1).cpu().numpy()
            #     target_probs = batch_actions.cpu().numpy()
                
            #     # 随机选一个样本进行详细对比 (也可以写死 idx = 0)
            #     idx = np.random.randint(0, pred_probs.shape[0]) 
                
            #     pred_p = pred_probs[idx]
            #     target_p = target_probs[idx]
                
            #     top_k = 5
            #     # 找到网络预测的 Top-K
            #     pred_top_indices = np.argsort(pred_p)[-top_k:][::-1]
            #     # 找到 MCTS 目标的 Top-K
            #     target_top_indices = np.argsort(target_p)[-top_k:][::-1]
                
            #     print(f"\n--- 样本 {idx} Policy 对比 ---")
            #     print(f"{'排名':<4} | {'网络预测动作 (位置)':<20} | {'MCTS目标动作 (位置)':<20}")
            #     print("-" * 60)
            #     for rank in range(top_k):
            #         p_act = pred_top_indices[rank]
            #         t_act = target_top_indices[rank]
            #         p_row, p_col = divmod(p_act, self.board_size)
            #         t_row, t_col = divmod(t_act, self.board_size)
                    
            #         p_val = pred_p[p_act]
            #         t_val = target_p[t_act]
                    
            #         print(f"{rank+1:<4} | {p_val:.4f} (坐标:{p_row},{p_col})  | {t_val:.4f} (坐标:{t_row},{t_col})")
                
            #     # 计算这个样本的 KL 散度 (衡量两个分布的差异，0代表完全一致)
            #     kl_div = np.sum(target_p * (np.log(target_p + 1e-10) - np.log(pred_p + 1e-10)))
            #     print(f"样本 {idx} KL散度: {kl_div:.4f}")
                
            #     # 额外检查：MCTS 的最佳动作，在网络眼中排第几？
            #     best_mcts_action = target_top_indices[0]
            #     rank_in_pred = np.where(np.argsort(pred_p)[::-1] == best_mcts_action)[0][0] + 1
            #     print(f"MCTS最佳动作 {best_mcts_action} 在网络预测中排第 {rank_in_pred} 位\n")

            # ============ 诊断结束 ============
            # with torch.no_grad():
            #     # 取一个样本
            #     sample_state  = batch_states[idx].cpu().numpy()   # (C,15,15)
            #     sample_target = batch_actions[idx].cpu().numpy()  # (225,)
                
            #     # 计算网络的预测概率
            #     sample_logits = p_logits[idx].cpu().numpy()
            #     sample_pred   = F.softmax(torch.tensor(sample_logits), dim=0).numpy() # (225,)
                
            #     # 从state还原棋盘
            #     board_cur = sample_state[0]   
            #     board_opp = sample_state[1]   
            #     board_2d  = board_cur - board_opp  # 1=当前方, -1=对手, 0=空
                
            #     # 1. 检测：膨胀Mask外的空位（孤岛）
            #     valid_mask = MCTS.get_valid_mask(board_2d.astype(np.int8), radius=2)
            #     target_nonzero = np.where(sample_target > 0)[0]
            #     outside_mask   = target_nonzero[valid_mask[target_nonzero] == 0]
                
            #     # 2. 检测：已有棋子的位置（绝对非法区）
            #     occupied_positions = np.where(board_2d.ravel() != 0)[0]
                
            #     # --- 综合计算 ---
            #     # MCTS 在已有棋子上的概率总和
            #     mcts_illegal_occupied_prob = np.sum(sample_target[occupied_positions])
            #     # MCTS 在膨胀Mask外的概率总和
            #     mcts_illegal_island_prob = np.sum(sample_target[valid_mask == 0])
                
            #     # 网络在已有棋子上的概率总和
            #     net_illegal_occupied_prob = np.sum(sample_pred[occupied_positions])
            #     # 网络在膨胀Mask外的概率总和
            #     net_illegal_island_prob = np.sum(sample_pred[valid_mask == 0])

            #     print(f"\n--- 样本 {idx} 综合合法性诊断 ---")
            #     print(f"棋盘已有棋子数: {len(occupied_positions)}")
                
            #     print("\n[1] 已有棋子位置 (绝对非法):")
            #     print(f"  MCTS 分配的概率总和: {mcts_illegal_occupied_prob:.6f}  (理想: 0)")
            #     print(f"  网络 分配的概率总和: {net_illegal_occupied_prob:.6f}  (理想: 0)")
            #     if mcts_illegal_occupied_prob > 0.001:
            #         print("  ❌ 严重: MCTS 把概率分给了已有棋子！搜索树有Bug！")
            #     elif net_illegal_occupied_prob > 0.1:
            #         print("  ⚠️ 警告: MCTS没教错，但网络依然把大量概率分给了已有棋子！")
                    
            #     print("\n[2] 膨胀Mask外的空位 (孤岛):")
            #     print(f"  MCTS 分配的概率总和: {mcts_illegal_island_prob:.6f}  (理想: 0)")
            #     print(f"  网络 分配的概率总和: {net_illegal_island_prob:.6f}  (理想: 0)")
            #     if net_illegal_island_prob > 0.5:
            #          print("  ⚠️ 警告: 网络在到处瞎下，把超过一半的概率分给了距离棋子很远的孤岛！")

            #     print("-" * 50)
            

            # 2. 构建绝对非法位置的 Mask (已有棋子的地方)
            # 假设 batch_states 的前两个通道是当前方和对手方
            # shape: (B, C, H, W)
            occupied_mask = (batch_states[:, 0:1, :, :] + batch_states[:, 1:2, :, :] > 0).float()
            occupied_mask = occupied_mask.view(-1, self.board_size * self.board_size) # (B, 225)

            # 3. 构建孤岛位置的 Mask (膨胀范围外的空位)
            # 注意：这里需要对 batch 中的每个棋盘计算 mask
            # 如果 get_valid_mask 计算较慢，可以只惩罚 occupied_mask，效果已经足够好
            # island_mask_list = []
            # for i in range(batch_states.shape[0]):
            #     board_cur = batch_states[i, 0].cpu().numpy()
            #     board_opp = batch_states[i, 1].cpu().numpy()
            #     board_2d = board_cur - board_opp
            #     valid_m = MCTS.get_valid_mask(board_2d.astype(np.int8), radius=2)
            #     # 孤岛 = 合法区取反
            #     island_m = 1.0 - valid_m 
            #     island_mask_list.append(island_m)
            #island_mask = torch.tensor(np.stack(island_mask_list), dtype=torch.float32, device=device)

            # 综合：绝对不能下的位置 = 已有棋子 OR 孤岛
            #illegal_mask = torch.clamp(occupied_mask + island_mask, min=0.0, max=1.0)

            legal_mask = 1.0 - occupied_mask
            masked_p_logits = p_logits.clone()
            masked_p_logits[legal_mask == 0] = -1e9 
            log_p = F.log_softmax(masked_p_logits, dim=1)
            pred_p = torch.exp(log_p)
            
            
            policy_loss_ce = -torch.mean(
                torch.sum(batch_actions * log_p, dim=1)
            )
            # 4. 计算网络在非法区域浪费的概率
            # pred_p shape: (B, 225)
            illegal_prob_sum = torch.sum(pred_p * occupied_mask, dim=1).mean()
            lambda_illegal = 10.0 
            policy_loss = policy_loss_ce + lambda_illegal * illegal_prob_sum
            total_loss = 5 * value_loss + policy_loss
            #print(f"value_loss: {value_loss.item():.3f} | policy_loss: {policy_loss.item():.3f}")
                        # =========================
            # entropy monitoring
            # =========================
            with torch.no_grad():
                # network policy probs
                pred_probs = F.softmax(masked_p_logits, dim=1)

                # network entropy
                pred_entropy = -(
                    pred_probs * torch.log(pred_probs + 1e-10)
                ).sum(dim=1).mean()

                # MCTS target entropy
                target_entropy = -(
                    batch_actions * torch.log(batch_actions + 1e-10)
                ).sum(dim=1).mean()

                # ================= 新增诊断代码 =================
                # 1. 计算 target (MCTS) 的动作数量和 Top-5
                # batch_actions 形状是 (B, 225)
                target_nonzero_counts = (batch_actions > 1e-6).sum(dim=1).float().mean() # 平均有多少个动作被赋予概率
                target_top5_vals, _ = torch.topk(batch_actions, k=5, dim=1)
                target_top5_mean = target_top5_vals.mean(dim=0) # 在 batch 维度求平均，得到 Top1-5 的平均概率

                # 2. 计算 Pred (网络预测) 的动作数量和 Top-5
                pred_nonzero_counts = (pred_probs > 1e-6).sum(dim=1).float().mean()
                pred_top5_vals, _ = torch.topk(pred_probs, k=5, dim=1)
                pred_top5_mean = pred_top5_vals.mean(dim=0)
                # ==================================================

            # 修改打印格式
            print(
                f"p_loss={policy_loss_ce.item():.4f} | "
                f"v_loss={value_loss.item():.4f} | "
                f"pred_entropy={pred_entropy.item():.4f} | "
                f"tar_entropy={target_entropy.item():.4f} | "
                f"illegal={illegal_prob_sum.item():.4f} | "
                
                # 打印动作数量
                f"pred_cnt={pred_nonzero_counts.item():.1f} | "
                f"tar_cnt={target_nonzero_counts.item():.1f} | "
                
                # 打印 Top-5 概率 (格式化更易读)
                f"pred_top5={[f'{v:.3f}' for v in pred_top5_mean.cpu().numpy()]} | "
                f"tar_top5={[f'{v:.3f}' for v in target_top5_mean.cpu().numpy()]}"
            )
            self.optimizer.zero_grad()
            total_loss.backward()
            #cursor 
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
        return value_loss.item(),policy_loss.item() 
    
       # print("Memory cleared，开始自对弈")
    def overfit_test(self, memory, steps=2000,batch_size=128):

        
        """过拟合测试：用固定batch反复训练，验证网络能否收敛"""
        print("\n========== 过拟合测试开始 ==========")
        
        all_transitions = list(memory.memory)  # 取所有数据
        pos = [t for t in all_transitions if t[2] == 1]
        neg = [t for t in all_transitions if t[2] == -1]
        
        print(f"正样本: {len(pos)}, 负样本: {len(neg)}")
        
        # 各取一半
        #self.network.eval()
        half = batch_size // 2
        import random
        balanced =  memory.sample(batch_size)
        # 在 overfit_test 开始时加
        print("Target policy 非零统计:")
        for i, (s, a, v) in enumerate(balanced[:5]):
            nonzero = np.count_nonzero(a)
            top3 = np.argsort(a)[-3:][::-1]
            print(f"样本{i}: 非零动作数={nonzero}, top3值={a[top3]}, top3位置={top3}")
        # self.network.eval()
        #balanced = memory.sample(16)
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
            # for name, p in self.network.named_parameters():
            #     if 'value' in name and p.grad is not None:
            #         print(f"{name}: grad_norm={p.grad.norm():.6f}")
            
            test_optimizer.step()
            if step % 100 == 0:
                # print(f"V 预测值样例: {v[:5].detach().cpu().numpy().flatten()}") # 打印前5个预测值
                # print(f"V 真实值样例: {batch_values[:5].cpu().numpy().flatten()}")
                #print(self.network.conv_input.weight.grad.norm())
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

    def get_winning_moves(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        向量化找必杀点（复用minimax里的逻辑），返回线性索引数组。
        预计算WINDOWS已在minimax模块里，直接import复用，零额外开销。
        """
        from minimax import WINDOWS  # 预计算好的5格窗口，直接复用
        flat = board.ravel()
        opp  = -player
        wv   = flat[WINDOWS]                          # (N_WIN, 5)
        p_cnt = np.sum(wv == player, axis=1)
        e_cnt = np.sum(wv == 0,      axis=1)
        o_cnt = np.sum(wv == opp,    axis=1)
        valid = (p_cnt == 4) & (e_cnt == 1) & (o_cnt == 0)
        if not valid.any():
            return np.array([], dtype=np.int32)
        vw  = WINDOWS[valid]
        vwv = wv[valid]
        ei  = np.argmax(vwv == 0, axis=1)
        return np.unique(vw[np.arange(len(vw)), ei])   # 去重后的必杀点
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
    # parser.add_argument('--board-size', type=int, default=15,
    #                    help='棋盘大小 (默认: 15)')
    args = parser.parse_args()
    agent = Agent(args.hyperparameters,args.train)

    
    if args.train:
        agent.run(is_training=True,render=False)
    else:
        agent.run(is_training=False,render=True)
    # if args.mode == 'gui':
    #     play_gui()
    # elif args.mode == 'test':
    #     test_env()

if __name__ == "__main__":
    main()
