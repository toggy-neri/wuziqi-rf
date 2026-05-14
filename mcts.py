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
        original_history = self.env.history.copy() if hasattr(self.env, 'history') else []
        policy = []
        if node.is_visited and node.actions is not None:
            noise = np.random.dirichlet([0.3] * len(node.actions))
            node.probs = 0.85 * node.probs + 0.15 * noise
       
        #expand the root node,get the node info
        for _ in range(search_num // batch_size):

            #expand the layer,use batch
            leaf_nodes = []
            leaf_states = []
            paths = []

            #depth first search the tree

            for _ in range(batch_size):
                path_len = 0
                #find the leaf node
                cur_node = node
                #t0 = time.time()
                path = [cur_node]
                #new_state = state.copy()


                #expand the tree
                
                while cur_node.is_visited and not cur_node.is_terminal:
                    action, next_node = self.get_best_child(
                        cur_node,
                        self.env.board,
                        exploration_factor
                    )

                    board_before = self.env.board.copy()

                    _, reward, done, info = self.env.step(action)

                    if info.get("invalid_move", False):

                        print("INVALID:", action)

                        self.env.undo()   # 关键！！！

                        # 不更新 node
                        continue

                    # ✔ valid move 才前进
                    cur_node = next_node
                    path.append(cur_node)
                    path_len += 1

                done, winner = self.env._check_win(self.env.last_move)
                v_terminal = None

                if done:
                    cur_node.is_terminal = True
                    v_terminal = 1 if winner == self.env.current_player else -1
                    


                leaf_nodes.append(cur_node)
                leaf_states.append((self.env.board.copy(),self.env.last_move,-self.env.current_player))
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
                
            #t1 = time.time()

            states_np = np.stack([self.env.get_channel_state(board,last_move,current_player) for board,last_move,current_player in leaf_states])
            inputs = torch.from_numpy(states_np).to(device, non_blocking=True).float()
            #inputs = torch.stack([torch.tensor(s.get_channel_state(s.board)) for s in leaf_states]).to(device)
            #done,winner = state._check_win(state.last_move)
            with torch.no_grad():
                policy , v = self.network(inputs)
            #t2 = time.time()
            
            #print(f"树搜索+copy: {t1-t0:.3f}s  |  网络推理: {t2-t1:.3f}s")
            #ta = time.time()
            # 从 leaf_states 提取 board 数组
            boards = np.stack([s for s,_,_ in leaf_states])  # (batch_size, 15, 15)
            probs = self.mask_and_softmax_batch(policy)
            #tb = time.time()
            probs = probs.cpu().numpy()

            # top_probs,top_actions = torch.topk(probs,30,dim =1 )
            # top_probs = top_probs.cpu().numpy()
            # top_actions = top_actions.cpu().numpy()
            #tc = time.time()
            #print(f"probs: {tb-ta:.4f}s | topk: {tc-tb:.4f}s")
            



            for i,(leaf,path) in enumerate(zip(leaf_nodes,paths)):
                #ta = time.time()
                #get the top n moves
                real_v = v_terminal if (path[-1].is_terminal and v_terminal is not None) else v[i].item()
                #tb = time.time()
                leaf.actions = np.arange(probs.shape[1])   # 所有动作索引
                leaf.probs = probs[i]                      # 对应概率
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
        self.env.last_move = original_last_move
        self.env.current_player = original_player
        if hasattr(self.env, 'history'):
            self.env.history = original_history


        
        #for child in root.children:
    def get_policy(self,node,board_size):
            total_policy = []
            #print(f"root children数: {len(node.children)}, visits: {node.visits}")
            p = np.zeros((board_size*board_size))
            for action,child in node.children.items():  #todo 混乱
                #scoreprint(f"  action={action}, visits={child.visits}, score={child.score:.3f}, Q={child.score/max(child.visits,1):.3f}")
                p[action] = child.visits
            sum_visits = p.sum()
            if sum_visits > 0:
                p = p / sum_visits

            return p
            

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
            if node.is_terminal:
                node.is_terminal = False

    def choose(self,root_node,is_training):
        actions = []
        visits = []

        for action,child in root_node.children.items():
            actions.append(action)
            visits.append(child.visits)

        if not is_training:
            best_action = np.argmax(visits)
            action = actions[best_action]
            return action,root_node.children[action]
        
        else:
            temperature = 1
            visits = np.array(visits)
            n_pow = np.power(visits, 1.0 / temperature)
            probs = n_pow / np.sum(n_pow)
            # 0 is pass
        action = np.random.choice(actions, p=probs)
        return action,root_node.children[action]



    def mask_and_softmax_batch(self, policy_logits_batch):

        probs = F.softmax(policy_logits_batch, dim=1)

        return probs

    def rollout(self, node):
        return #policy / policy.sum()

    def rollout(self, node):
        pass



    
    def get_best_child(self, node, board, c_puct=1.5):

        actions = node.actions
        ps = node.probs

        # 合法位置 mask
        legal_mask = (board.reshape(-1) == 0)

        # 只保留合法动作
        legal_actions = actions[legal_mask[actions]]
        legal_ps = ps[legal_mask[actions]]

        # 没有合法动作
        if len(legal_actions) == 0:
            return None, None

        # visits / scores
        vs = np.array([
            node.children[a].visits if a in node.children else 0
            for a in legal_actions
        ])

        ss = np.array([
            node.children[a].score if a in node.children else 0
            for a in legal_actions
        ])

        # PUCT
        q = np.where(vs > 0, ss / (vs + 1e-6), 0.0)

        u = c_puct * legal_ps * (
            np.sqrt(node.visits + 1e-8) / (1.0 + vs)
        )

        scores = q + u

        best_idx = np.argmax(scores)

        best_action = legal_actions[best_idx]

        # expand
        if best_action not in node.children:

            node.children[best_action] = TreeNode(parent=node)

            node.children[best_action].p_value = legal_ps[best_idx]

        child = node.children[best_action]

        # virtual loss
        v_loss = 1

        child.visits += v_loss
        child.score -= v_loss

        return best_action, child

