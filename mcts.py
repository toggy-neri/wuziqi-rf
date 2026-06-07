import torch
import torch.nn.functional as F
import numpy as np

#tree node definition
class TreeNode():
    def __init__(self, parent=None):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.score = 0
        self.p_value = 0
        self.is_terminal = False
        self.is_visited = False

        self.actions = None
        self.probs = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#MCTS algorithm
class MCTS():
    def __init__(self, network,env,is_training=True,use_virtual_loss=False,virtual_loss=1.0):
        self.network = network
        self.is_training = is_training
        self.env = env
        self.use_virtual_loss = use_virtual_loss
        self.virtual_loss = virtual_loss
    def search(self,node :TreeNode,batch_size=64,search_num=512,exploration_factor=1.5):
        if node is None:
            print("[MCTS] search got None root, skip this search")
            return
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

            need_network_indices = []   # 记录哪些叶节点需要走网络
            rule_probs_cache     = {}   # {叶节点在列表中的index: rule_probs}   
                #depth first search the tree

            for _ in range(batch_size):
                path_len = 0
                #find the leaf node
                cur_node = node
                path = [cur_node]
                v_terminal = None
                done = False

                
                #expand the tree
                
                while cur_node.is_visited :
                    is_root = (cur_node is node)
                    action, next_node = self.get_best_child(
                        cur_node,
                        is_root,
                        exploration_factor
                    )
                    if next_node is None:
                        break

                    if self.use_virtual_loss:
                        self.apply_virtual_loss(next_node)

                    _, reward, done, info = self.env.step(action)
                    path_len += 1
                    if done:
                        cur_node = next_node
                        path.append(next_node)
                        winner = info.get("winner", None)
                        v_terminal = 0 if winner == 0 else -1
                        break



                    #     # 不更新 node
                    

                    cur_node = next_node
                    path.append(cur_node)


                terminal_dones.append(done)
                terminal_values.append(v_terminal)    


                leaf_nodes.append(cur_node)
                leaf_states.append(self.env.get_channel_state(self.env.board, self.env.current_player))
                leaf_boards.append(self.env.board.copy())
                paths.append(path)

                current_idx = len(leaf_nodes) - 1
                board_now     = self.env.board
                cur_player    = self.env.current_player
                if done:
                    # 情况A：已终局，不需要网络，不需要规则
                    terminal_dones[current_idx] = True
                    terminal_values[current_idx] = v_terminal
                    # 占位符，后续 expand 会处理 uniform
                    rule_probs_cache[current_idx] = None 
                
                elif self.is_training:
                    # 情况B：未终局，检查是否有杀棋
                    # 这里检查的是 "当前轮到谁下，谁有没有杀招"
                    win_moves = self.get_winning_moves(board_now, cur_player)
                    
                    if len(win_moves) > 0:
                        # ✅ 命中规则：有杀棋
                        #int(f"   [Rule] Killing move found for Player {cur_player}")
                        
                        # --- 关键修复：将坐标转换为概率数组 ---
                        rule_prob = np.zeros(225, dtype=np.float32)
                        rule_prob[win_moves] = 1.0  # 所有杀招位置概率为 1
                        rule_prob = rule_prob / rule_prob.sum() # 归一化（如果有多个杀招）
                        
                        rule_probs_cache[current_idx] = rule_prob
                        terminal_dones[current_idx] = True # 视为确定性节点
                        terminal_values[current_idx] = 1.0 # 强行设定 Value 为 1
                    else:
                        # 情况C：没有杀棋，需要网络评估
                        terminal_dones[current_idx] = False
                        # --- 关键修复：只有在需要网络时才添加到 leaf_states ---
                        need_network_indices.append(current_idx)
                else:
                    terminal_dones[current_idx] = False
                    need_network_indices.append(current_idx)


                for _ in range(path_len):
                    self.env.undo()
                
            network_probs = {}   # {index: prob_array}
            network_values = {}  # {index: value}
            if need_network_indices:
                net_states = np.stack([leaf_states[i] for i in need_network_indices])
                inputs = torch.from_numpy(net_states).to(device, non_blocking=True).float()
                with torch.no_grad():
                    policy_logits, v_net = self.network(inputs)
                probs_batch = self.mask_and_softmax_batch(policy_logits).cpu().numpy()

                for batch_pos, leaf_idx in enumerate(need_network_indices):
                    board_i  = leaf_boards[leaf_idx]
                    raw_prob = probs_batch[batch_pos]

                    # 应用valid_mask
                    valid_mask = self.get_valid_mask(board_i, radius=2)
                    raw_prob   = raw_prob * valid_mask
                    prob_sum   = raw_prob.sum()
                    if prob_sum > 1e-8:
                        raw_prob = raw_prob / prob_sum
                    else:
                        raw_prob = np.zeros_like(raw_prob)

                    network_probs[leaf_idx]  = raw_prob
                    network_values[leaf_idx] = v_net[batch_pos].item()



            
            



            for i,(leaf,path) in enumerate(zip(leaf_nodes,paths)):
                if terminal_dones[i] and i not in need_network_indices:
                # 终局 or 规则命中
                    real_v = terminal_values[i]
                    if i in rule_probs_cache and rule_probs_cache[i] is not None:
                        # 规则给的probs直接用
                        final_probs = rule_probs_cache[i]
                    else:
                        # 真终局：uniform fallback
                        valid_mask  = self.get_valid_mask(leaf_boards[i], radius=2)
                        final_probs = valid_mask / (valid_mask.sum() + 1e-8)
                else:
                    # 网络给的结果
                    real_v      = network_values.get(i, 0.0)
                    final_probs = network_probs.get(i, None)
                    if final_probs is None:
                        valid_mask  = self.get_valid_mask(leaf_boards[i], radius=2)
                        final_probs = valid_mask / (valid_mask.sum() + 1e-8)
                if final_probs is not None:
                    if len(final_probs) == 0:
                        print(f"final_probs is None, i={i}")

                if final_probs is None:
                    print(f"final_probs is empty, i={i}")

                self.expand_node(
                    leaf,
                    leaf_boards[i],
                    final_probs,
                    min_prob=0.0 if terminal_dones[i] else 1e-12
                ) 

                if(len(leaf.actions) == 0):
                    break
                    raise ValueError("leaf.actions is empty")
                leaf.is_visited = True

                self.backpropagate(path, real_v)
       

        self.env.board = original_board
        self.env.last_move = original_last_move
        self.env.current_player = original_player
        if hasattr(self.env, 'move_memory'):
            self.env.move_memory = original_history


        
    def get_policy(self,node,board_size):
            p = np.zeros((board_size*board_size))
            for action,child in node.children.items():
                p[action] = child.visits
            sum_visits = p.sum()
            if sum_visits > 0:
                p = p / sum_visits

            return p
    
    def expand_node(self, node: TreeNode, board: np.ndarray, raw_probs: np.ndarray, min_prob: float = 0.0):
        """在节点首次被访问时调用，只提取并保存合法动作及其概率"""
        # 1. 获取当前棋盘的合法位置 Mask
        valid_mask = self.get_valid_mask(board, radius=2)
        
        # 2. 找到所有合法动作的索引
        if min_prob > 0:
            valid_actions = np.where((valid_mask > 0) & (raw_probs > min_prob))[0]
        else:
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


            

    def backpropagate(self, path, value):
        score = value
        for node in reversed(path):  
            if self.use_virtual_loss and node.parent is not None:
                node.visits -= 1
                node.score -= self.virtual_loss
            
            node.visits += 1
            node.score += score
            score = -score

    def apply_virtual_loss(self, node):
        node.visits += 1
        node.score += self.virtual_loss

    def choose(self, root_node, is_training, current_step=0):
            if root_node is None:
                action = self.get_fallback_action()
                print(f"[MCTS] choose got None root, fallback action={action}")
                return action, TreeNode(parent=None)
            actions = []
            visits = []
            valid_mask = self.get_valid_mask(self.env.board, radius=2)
            legal_children = {
                a: child for a, child in root_node.children.items() 
                if valid_mask[a] > 0
            }
            for action, child in legal_children.items():
                actions.append(action)
                visits.append(child.visits)
                
            if len(actions) == 0:
                action = self.get_fallback_action()
                print(f"[MCTS] choose found no legal children, fallback action={action}")
                return action, TreeNode(parent=None)
                
            visits = np.array(visits, dtype=np.float64)
            
            if not is_training:
                # 测试/对战时：直接选访问次数最多的
                best_action = np.argmax(visits)
                return actions[best_action], legal_children[actions[best_action]]
            
            else:
                # 训练时：使用标准温度策略
                # 开局前 10 步 (可根据棋盘大小调整，15x15 推荐 10-15步)，使用 tau=1.0 鼓励探索
                if current_step < 10:
                    temperature = 1.0
                else:
                    # 10 步之后，使用极小的 tau (趋近于 0)，等价于 argmax，保证对局质量
                    temperature = 1e-3 
                
                # 防止数值溢出
                if temperature > 0.01:
                    n_pow = np.power(visits + 1e-10, 1.0 / temperature)
                    probs = n_pow / np.sum(n_pow)
                    action = np.random.choice(actions, p=probs)
                else:
                    # tau 极小时，直接取 argmax
                    action = actions[np.argmax(visits)]
                    
                return action, legal_children[action]


    def get_fallback_action(self):
        valid_mask = self.get_valid_mask(self.env.board, radius=2)
        actions = np.where(valid_mask > 0)[0]
        if len(actions) == 0:
            actions = np.where(self.env.board.ravel() == 0)[0]
        if len(actions) == 0:
            raise ValueError("[MCTS] no valid fallback actions")
        return int(np.random.choice(actions))


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
            
            r_start = max(0, center - radius)
            r_end = min(size, center + radius + 1)
            
            # 将中心的 5x5 区域置为1
            mask[r_start:r_end, r_start:r_end] = 1.0
            
            # 展平为一维数组并返回
            return mask.ravel()
        
        # 向量化：用最大池化膨胀occupied区域
        occupied = (board != 0)
        # 生成radius大小的结构元素
        dilated = np.zeros_like(occupied, dtype=bool)
        struct = None
        def binary_dilation(occupied_board, structure=None):
            dilated_board = np.zeros_like(occupied_board, dtype=bool)
            occupied_rows, occupied_cols = np.where(occupied_board)
            for row, col in zip(occupied_rows, occupied_cols):
                r_start = max(0, row - radius)
                r_end = min(size, row + radius + 1)
                c_start = max(0, col - radius)
                c_end = min(size, col + radius + 1)
                dilated_board[r_start:r_end, c_start:c_end] = True
            return dilated_board
        dilated = binary_dilation(occupied, structure=struct)  # 膨胀
        valid_2d = dilated & (board == 0)
        return valid_2d.ravel().astype(np.float32)
    def rollout(self, node):
        pass
    
    
    def _get_winning_moves_windows_15(self, board: np.ndarray, player: int) -> np.ndarray:
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
        return np.unique(vw[np.arange(len(vw)), ei]) 


    def get_winning_moves(self, board: np.ndarray, player: int) -> np.ndarray:
        size = board.shape[0]
        winning_moves = []
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))

        for action in np.flatnonzero(board.ravel() == 0):
            row, col = divmod(int(action), size)
            for dr, dc in directions:
                count = 1

                r, c = row + dr, col + dc
                while 0 <= r < size and 0 <= c < size and board[r, c] == player:
                    count += 1
                    r += dr
                    c += dc

                r, c = row - dr, col - dc
                while 0 <= r < size and 0 <= c < size and board[r, c] == player:
                    count += 1
                    r -= dr
                    c -= dc

                if count >= 5:
                    winning_moves.append(int(action))
                    break

        return np.array(winning_moves, dtype=np.int32)

    def get_best_child(self, node,  is_root,c_puct=1.5):
        c_puct = 3.0 if is_root else c_puct
    
        actions = node.actions
        ps = node.probs

        # 合法位置 mask


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
        q = np.where(vs > 0, -ss / (vs + 1e-6), 0.0)
        u = c_puct * ps * (
            np.sqrt(max(node.visits,0) + 1e-8) / (1.0 + vs)
        )

        scores = q + u

        best_idx = np.argmax(scores)

        best_action = actions[best_idx]

        # expand
        if best_action not in node.children:

            node.children[best_action] = TreeNode(parent=node)


        child = node.children[best_action]



        return best_action, child
    def debug_tree(self, node, depth=0):
        indent = "  " * depth

        print(
            f"{indent}N={node.visits} "
            f"W={node.score:.2f}"
        )

        for action, child in node.children.items():

            q = child.score / child.visits if child.visits > 0 else 0

            print(
                f"{indent}├─ action={action} "
                f"N={child.visits} "
                f"W={child.score:.2f} "
                f"Q={q:.2f}"
            )

            self.debug_tree(child, depth+1)
