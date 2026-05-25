#
#MCTS ALGORITHM
#
import copy
import math
import random
from network import Network, Residual_block
from wuziqi_env import WuziqiEnv
import torch
import torch.nn.functional as F
import numpy as np
import time

#tree node definition
class TreeNode():
    #class init function
    def __init__(self, parent=None):
        self.parent = parent
        #self.action = action
        self.children = {}
        self.visits = 0
        self.score = 0
        self.p_value = 0
        #self.w = 0
        self.is_terminal = False
        self.is_visited = False

        self.actions = None
        self.probs = None
#        self.children , self.score = self.network(self.state)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#MCTS algorithm
class MCTS():
    def __init__(self, network,env,is_training=True):
        self.network = network
        self.is_training = is_training
        self.env = env
    def search(self,node :TreeNode,batch_size=64,search_num=512,exploration_factor=1.5):
        #create root node 
        original_board = self.env.board.copy()
        original_last_move = self.env.last_move
        original_player = self.env.current_player
        original_history = self.env.move_memory.copy() if hasattr(self.env, 'move_memory') else []
        policy = []
        if node.is_visited and node.actions is not None and self.is_training:
            noise = np.random.dirichlet([0.044] * len(node.actions))
            node.probs = 0.75 * node.probs + 0.25 * noise
       
        #expand the root node,get the node info
        for _ in range(search_num // batch_size):

            #expand the layer,use batch
            leaf_nodes = []
            leaf_states = []
            leaf_boards = [] 
            paths = []
            terminal_dones = []
            terminal_values = []
            #depth first search the tree

            for _ in range(batch_size):
                path_len = 0
                #find the leaf node
                cur_node = node
                # t0 = time.time()
                path = [cur_node]
                #new_state = state.copy()
                v_terminal = None
                done = False

                
                #expand the tree
                
                while cur_node.is_visited :
                    is_root = (cur_node is node)
                    action, next_node = self.get_best_child(
                        cur_node,
                        self.env.board,
                        is_root,
                        exploration_factor
                    )
                    #self.env.current_player = -self.env.current_player
                    if next_node is None:
                        break

                    _, reward, done, info = self.env.step(action)
                    if done:
                        v_terminal = 1 if info["winner"] == -self.env.current_player else -1
                        break
                    # if info.get("invalid_move", False):

                    #     print("INVALID:", action)

                    #     self.env.undo()   # 关键！！！

                    #     # 不更新 node
                    #     continue
                    

                    # ✔ valid move 才前进
                    cur_node = next_node
                    path.append(cur_node)
                    path_len += 1

                #done, winner = self.env._check_win(self.env.last_move)

                terminal_dones.append(done)
                terminal_values.append(v_terminal)    


                leaf_nodes.append(cur_node)
                leaf_states.append(self.env.get_channel_state(self.env.board, self.env.current_player))
                leaf_boards.append(self.env.board.copy())
                paths.append(path)

                for _ in range(path_len):
                    self.env.undo()
                #undo the path,aviod too much memory usage
                # try:
                #     for _ in range(path_len):
                #         print(self.env.board)
                #         print(path_len)
                #         print(_)
                #         self.env.undo()
                # except Exception as e:
                #     print("UNDO ERROR:", e)
                #     pass
                
            # t1 = time.time()

            states_np = np.stack(leaf_states)
            inputs = torch.from_numpy(states_np).to(device, non_blocking=True).float()
            #inputs = torch.stack([torch.tensor(s.get_channel_state(s.board)) for s in leaf_states]).to(device)
            #done,winner = state._check_win(state.last_move)
            # with torch.no_grad():
            #     policy , v = self.network(inputs)
            # valid_mask = self.get_valid_mask(self.env.board, radius=2)

            # valid_mask_tensor = torch.tensor(valid_mask, dtype=policy.dtype, device=policy.device)

            # 张量相乘
            # policy = policy * valid_mask_tensor
            with torch.no_grad():
                policy , v = self.network(inputs)
            # 生成蒙版并应用
            policy = self.mask_and_softmax_batch(policy)
            valid_mask = self.get_valid_mask(self.env.board, radius=2)
            valid_mask_tensor = torch.tensor(valid_mask, dtype=policy.dtype, device=policy.device)
            policy = policy * valid_mask_tensor                    # 非法位置概率清零
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum                # 重新归一化
            else:
                # 极端情况：蒙版全0，fallback到均匀分布
                policy = valid_mask / valid_mask.sum()
            # t2 = time.time()
            
            # print(f"树搜索+copy: {t1-t0:.3f}s  |  网络推理: {t2-t1:.3f}s")
            #ta = time.time()
            # 从 leaf_states 提取 board 数组
            #boards = np.stack([s for s,_ in leaf_states])  # (batch_size, 15, 15)
            probs = self.mask_and_softmax_batch(policy)
            #tb = time.time()
            probs = probs.cpu().numpy()
            if(len(probs) == 0):
                raise ValueError("probs is empty")
            # top_probs,top_actions = torch.topk(probs,30,dim =1 )
            # top_probs = top_probs.cpu().numpy()
            # top_actions = top_actions.cpu().numpy()
            #tc = time.time()
            #print(f"probs: {tb-ta:.4f}s | topk: {tc-tb:.4f}s")
            



            for i,(leaf,path) in enumerate(zip(leaf_nodes,paths)):
                #ta = time.time()
                #get the top n moves
                if terminal_dones[i]:
                    real_v = terminal_values[i]
                else:
                    real_v = v[i].item()
                #tb = time.time()
                # leaf.actions = np.arange(probs.shape[1])   # 所有动作索引
                # leaf.probs = probs[i]                      # 对应概率

                self.expand_node(leaf, leaf_boards[i], probs[i]) 

                if(len(leaf.actions) == 0):
                    raise ValueError("leaf.actions is empty")
                #print(leaf.actions)
                #print(leaf.probs)
                # #update the p_dict
                # leaf.actions = top_actions[i]
                # leaf.probs = top_probs[i]
                #tc = time.time()
                leaf.is_visited = True

                self.backpropagate(path, real_v)
       
                #td = time.time()
            #print(f"topk: {tb-ta:.4f}s | dict: {tc-tb:.4f}s | backprop: {td-tc:.4f}s")
            # t3 = time.time()
            # t_tree  = t1 - t0
            # t_infer = t2 - t1
            # t_back  = t3 - t2

            #print(f"树搜索: {t_tree:.3f}s | 推理: {t_infer:.3f}s | 反传+赋值: {t_back:.3f}s")
        self.env.board = original_board
        self.env
        self.env.last_move = original_last_move
        self.env.current_player = original_player
        if hasattr(self.env, 'move_memory'):
            self.env.move_memory = original_history


        
        #for child in root.children:
    def get_policy(self,node,board_size):
            total_policy = []
            #print(f"root children数: {len(node.children)}, visits: {node.visits}")
            p = np.zeros((board_size*board_size))
            for action,child in node.children.items():  #todo 混乱
                #scoreprint(f"  action={action}, visits={child.visits}, score={child.score:.3f}, Q={child.score/max(child.visits,1):.3f}")
                p[action] = child.visits
            sum_visits = p.sum()
            #print(p)
            if sum_visits > 0:
                p = p / sum_visits

            return p
    
    def expand_node(self, node: TreeNode, board: np.ndarray, raw_probs: np.ndarray):
        """在节点首次被访问时调用，只提取并保存合法动作及其概率"""
        # 1. 获取当前棋盘的合法位置 Mask
        valid_mask = self.get_valid_mask(board, radius=2)
        
        # 2. 找到所有合法动作的索引
        valid_actions = np.where(valid_mask > 0)[0]
        
        # 【防御1】：如果物理上就没有合法位置（理论上不应发生，除非终局判断漏了）
        if len(valid_actions) == 0:
            node.actions = np.array([], dtype=np.int64)
            node.probs = np.array([], dtype=np.float32)
            return

        # 【防御2】：拦截 NaN 和无穷大
        if np.any(np.isnan(raw_probs)) or np.any(np.isinf(raw_probs)):
            print("⚠️ 警告：网络输出包含 NaN 或 Inf！回退到均匀分布")
            valid_probs = np.ones(len(valid_actions), dtype=np.float32) / len(valid_actions)
            node.actions = valid_actions
            node.probs = valid_probs
            return

        # 3. 提取合法动作对应的概率
        valid_probs = raw_probs[valid_actions].copy()
        
        # 【防御3】：截断负数（防止网络输出极小的负数导致求和异常）
        valid_probs = np.maximum(valid_probs, 0)

        # 4. 重新归一化
        prob_sum = valid_probs.sum()
        if prob_sum > 1e-8:  # 提高浮点数比较的安全阈值
            valid_probs = valid_probs / prob_sum
        else:
            # 【防御4】：如果网络把所有合法位置的概率都预测成了 0，回退到均匀分布
            print("⚠️ 警告：网络对所有合法位置预测概率为0！回退到均匀分布")
            valid_probs = np.ones(len(valid_actions), dtype=np.float32) / len(valid_actions)

        # 5. 保存到节点中
        node.actions = valid_actions
        node.probs = valid_probs


            

                       #self.walk(self.root)
    def backpropagate(self, path, value):
        score = value
        v_loss = 1
        for node in reversed(path):  
            if node.parent is not None: 
                node.visits -= v_loss
                node.score += v_loss
            
            node.visits += 1
            node.score += score
            score = -score
            # if node.is_terminal:
            #     continue
            # if node.is_terminal:
            #     node.is_terminal = False

    def choose(self,root_node,is_training):
        actions = []
        visits = []
        valid_mask = self.get_valid_mask(self.env.board, radius=2)
        legal_children = {
        a: child for a, child in root_node.children.items() 
        if valid_mask[a] > 0
        }
        for action,child in legal_children.items():
            actions.append(action)
            visits.append(child.visits)
        if(len(actions) == 0):
            raise ValueError("actions is empty")
        if not is_training:
            best_action = np.argmax(visits)
            action = actions[best_action]
            return action,legal_children[action]
        
        else:
            temperature = 0.5
            visits = np.array(visits)
            n_pow = np.power(visits, 1.0 / temperature)
            probs = n_pow / np.sum(n_pow)
            # 0 is pass
        action = np.random.choice(actions, p=probs)
        return action,legal_children[action]



    def mask_and_softmax_batch(self, policy_logits_batch):


        probs = F.softmax(policy_logits_batch, dim=1)

        return probs
    @classmethod
    def get_valid_mask(self,board: np.ndarray, radius: int = 2) -> np.ndarray:
        size = board.shape[0]
        
        if not np.any(board != 0):
            # 棋盘为空时的逻辑
            mask = np.zeros((size, size), dtype=np.float32)
            
            # 计算中心的索引 (例如 15x15 棋盘，center为7)
            center = size // 2 
            
            # 计算 5x5 区域的边界索引 (radius=2 时，扩展2格)
            # r_start = 7 - 2 = 5, r_end = 7 + 3 = 10 (切片不包含10，即 5~9 共5个索引)
            r_start = max(0, center - radius)
            r_end = min(size, center + radius + 1)
            
            # 将中心的 5x5 区域置为1
            mask[r_start:r_end, r_start:r_end] = 1.0
            
            # 展平为一维数组并返回
            return mask.ravel()
        
        # 向量化：用最大池化膨胀occupied区域
        from scipy.ndimage import binary_dilation
        occupied = (board != 0)
        # 生成radius大小的结构元素
        struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        dilated = binary_dilation(occupied, structure=struct)  # 膨胀
        # 合法位置 = 膨胀后为True 且 当前为空
        valid_2d = dilated & (board == 0)
        return valid_2d.ravel().astype(np.float32)
    def rollout(self, node):
        return #policy / policy.sum()

    def rollout(self, node):
        pass



    
    def get_best_child(self, node, board, is_root,c_puct=1.5):
        c_puct = 2.0 if is_root else c_puct
    
        actions = node.actions
        ps = node.probs

        # 合法位置 mask
        # legal_mask = (board.reshape(-1) == 0)

        # # 只保留合法动作
        # legal_actions = actions[legal_mask[actions]]
        # legal_ps = ps[legal_mask[actions]]

        # 没有合法动作
        if len(actions) == 0:
            return None, None

        # visits / scores
        vs = np.array([
            node.children[a].visits if a in node.children else 0
            for a in actions
        ])

        ss = np.array([
            node.children[a].score if a in node.children else 0
            for a in actions
        ])
        #print(ss)
        # PUCT
        q = np.where(vs > 0, ss / (vs + 1e-6), 0.0)
        u = c_puct * ps * (
            np.sqrt(max(node.visits,0) + 1e-8) / (1.0 + vs)
        )

        scores = q + u

        best_idx = np.argmax(scores)

        best_action = actions[best_idx]

        # expand
        if best_action not in node.children:

            node.children[best_action] = TreeNode(parent=node)

            node.children[best_action].p_value = ps[best_idx]

        child = node.children[best_action]

       # virtual loss
        v_loss = 1

        child.visits += v_loss
        child.score -= v_loss

        return best_action, child

