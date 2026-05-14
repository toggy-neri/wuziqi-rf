from datetime import datetime
import sys
import argparse
import time
import yaml
import os

#chess GUI
from wuziqi_gui import WuziqiGUI,AiMatch

#chess control
from wuziqi_env import WuziqiEnv




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

        self.is_continue_training = hyperparameters['is_continue_training']
        #store the model file and info 
        self.exist_model_name = hyperparameters['exist_model_name']
        self.MODEL_FILE = os.path.join(self.run_dir, f'{self.exist_model_name}.pt')

        self.MODEL_FILE_RESTORE = os.path.join(self.restore_dir, f'{hyperparameters_set}.pt')
        self.LOG_FILE = os.path.join(self.restore_dir, f'{hyperparameters_set}.txt')
        self.LOG_FILE_OPTIMIZE = os.path.join(self.restore_dir, f'{hyperparameters_set}_optimize.txt')  
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
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            resore_count = 0
            #store path of the mddel
            os.makedirs(self.run_dir, exist_ok=True)
        else:
            ai_match = AiMatch(env, self.network,self.search_num,self.inference_batch_size)
            ai_match.run()

        #create network
 
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr,weight_decay=1e-5)#此处更换


        #create mcts
        mcts = MCTS(self.network,env,is_training=self.is_training)

        #update rate   
        sync_count = 0
        restore_count = 0

    

        
        #for i in range(self.self_play_num):
        for i in range(self.self_play_num):
            time_start = time.time()
            root = TreeNode(parent=None)
            game_history = []
            state = env.reset()
            done = False
            #flag = 0
            
            while not done:
                #flag += 1
                current_time = datetime.now()
                #print( f"Time: {current_time.strftime(DATE_FORMAT)}")
                
                #t_search = time.time()
                #search the tree,return policy(15*15)
                mcts.search(root,self.inference_batch_size,self.search_num,self.exploration_factor)
                #t1 = time.time()
                policy = mcts.get_policy(root,self.board_size)
                #choose the best move 
                action,root = mcts.choose(root,is_training)

                #t2 = time.time()
                root.parent = None
                #step forward
                state, reward, done, info = env.step(action)  
                #store the game history
                if is_training:
                    now_player = env.current_player
                    game_history.append((state.copy(),policy.copy(),-now_player,env.last_move))
                
                #t3 = time.time()
                
                #print(f"search: {t1-t_search:.3f}s | choose: {t2-t1:.3f}s | step+copy: {t3-t2:.3f}s | flag: {flag}")
                #print(env.board)
                #
                if done:

                    winner = info["winner"]

                    for s,p,player,last_move in game_history:  #注意不要保证重名
                        #print(s)
                        value = 1 if player == winner else -1
                        ch_state = env.get_channel_state(s,last_move,player)
                        print(f"Policy Max Prob: {np.max(p):.4f}, Non-zero actions: {np.count_nonzero(p)}")
                        memory.append((ch_state, p,value))
                        # if len(memory) >= self.replay_memory_size:
                        #     self.optimize(memory, self.optimizer_batch_size)


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
                total_loss = self.optimize(memory, self.optimizer_batch_size)
                print(f"current_time: {current_time},total_time: {current_time-start_time}, Epoch {restore_count},epoch_time: {time_end-time_start:.3f}s, Loss {total_loss:.4f}")


                if restore_count % self.restore_epoch == 0:
                    with open(self.LOG_FILE, 'a') as f:
                        f.write(f"current_time: {current_time},total_time: {current_time-start_time}, Epoch {restore_count},epoch_time: {time_end-time_start:.3f}s, Loss {total_loss:.4f}\n")
                        torch.save(self.network.state_dict(), self.MODEL_FILE_RESTORE)
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

            # # 旋转增强：对每个样本生成4个旋转版本
            # aug_states = []
            # aug_actions = []
            # aug_values = []

            # for k in range(4):  # k=0,1,2,3 对应 0°,90°,180°,270°
            #     # 旋转棋盘：state的后两个维度是棋盘(H,W)，对axis=(2,3)旋转
            #     rotated_states = np.rot90(states, k=k, axes=(2, 3)).copy()

            #     # policy也要跟着旋转：先reshape成棋盘形状，旋转，再展平
            #     rotated_actions = actions.reshape(-1, self.board_size, self.board_size)
            #     rotated_actions = np.rot90(rotated_actions, k=k, axes=(1, 2)).copy()
            #     rotated_actions = rotated_actions.reshape(-1, self.board_size * self.board_size)

            #     aug_states.append(rotated_states)
            #     aug_actions.append(rotated_actions)
            #     aug_values.append(values)

            # # 拼接成 4*B 的大batch
            # aug_states  = np.concatenate(aug_states,  axis=0)
            # aug_actions = np.concatenate(aug_actions, axis=0)
            # aug_values  = np.concatenate(aug_values,  axis=0)

            # batch_states  = torch.FloatTensor(aug_states).to(device)
            # batch_actions = torch.FloatTensor(aug_actions).to(device)
            # batch_values  = torch.FloatTensor(aug_values).to(device).view(-1, 1)
            batch_states  = torch.FloatTensor(states).to(device)
            batch_actions = torch.FloatTensor(actions).to(device)
            batch_values  = torch.FloatTensor(values).to(device).view(-1, 1)
            p_logits, v = self.network(batch_states)
            print(f"v mean: {v.mean().item():.3f}, std: {v.std().item():.3f}")
            print(f"target mean: {batch_values.mean().item():.3f}, std: {batch_values.std().item():.3f}")

            value_loss  = F.mse_loss(v, batch_values)
            log_p       = F.log_softmax(p_logits, dim=1)
            policy_loss = -torch.mean(torch.sum(batch_actions * log_p, dim=1))

            total_loss = 5 * value_loss + policy_loss
            print(f"value_loss: {value_loss.item():.3f} | policy_loss: {policy_loss.item():.3f}")

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        return total_loss.item() 

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
