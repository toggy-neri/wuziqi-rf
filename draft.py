    def search(self,node :TreeNode,state:WuziqiEnv,batch_size=4):
        #create root node 

        #check if the node is terminal node

        policy = []
        root = node
       
        #expand the root node,get the node info
        for _ in range(500 // batch_size):

            #expand the layer,use batch
            leaf_nodes = []
            leaf_states = []
            paths = []

            #depth first search the tree

            for _ in range(batch_size):
                path_len = 0
                #find the leaf node
                cur_node = node
                path = []

                #new_state = state.copy()


                #expand the tree
                while cur_node.is_visited:
                    action, cur_node = self.get_best_child(cur_node)
                    state.step(action)
                    path_len += 1
                    path.append((cur_node,action))

                leaf_nodes.append(cur_node)
                leaf_states.append(state)
                paths.append(path)
                
                #undo the path,aviod too much memory usage
                for _ in range(path_len):
                    state.undo()

            inputs = torch.stack([state.get_channel_state(state.board) for state in leaf_states]).to(device)
            #done,winner = state._check_win(state.last_move)
            with torch.no_grad():
                policy , v = self.network(inputs)

            for i,(leaf,state,path) in enumerate(zip(leaf_nodes,leaf_states,paths)):

                policy = self.mask_and_softmax(policy[i:i+1],state).squeeze(0)
                #get the top n moves
                top_probs,top_actions = torch.topk(policy, 30)

                #update the p_dict
                leaf.p_dict = {act.item(): prob.item() for act, prob in zip(top_actions, top_probs)}

                leaf.is_visited = True

                self.backpropagate(leaf, v.item())

            if not done:
                #node.children = {}
                            #expand the layer
                input = state.get_channel_state(state.board)
                input = torch.tensor(input).to(device)
                input = input.unsqueeze(0)

                if not self.is_training:
                    self.network.eval()

                with torch.no_grad():
                    policy , v = self.network(input)

                #mask the invalid moves
                policy = self.mask_and_softmax(policy,state)
                policy = policy.squeeze(0)
                top_probs,top_actions = torch.topk(policy, 30)#todo:hyperparameter search

                action_probs = {act.item(): prob.item() for act, prob in zip(top_actions, top_probs)}
                node.p_dict.update(action_probs)
                #expand the children node
                
                # for action,prob in zip(top_actions,top_probs):
                #     action = action.item()
                #     child_state = new_state.copy()
                #     child_state.step(action)

                #     node.children.update({action:TreeNode(parent=node)})  
                #     node.children[action].p_value = prob.item()
                node.is_visited = True
            else:
                v = 1 if winner == state.current_player else -1
                node.is_terminal = True
            self.backpropagate(node, v.item())

            for _ in range(path_len):
                state.undo()

        def mask_and_softmax(self, policy_logits, state):
        """
        policy_logits: 神经网络输出的原始值 (1, 225)
        state: 当前棋盘环境
        """
    
        valid_moves = state.get_valid_moves()
        
        mask = torch.full(policy_logits.shape, -1e10).to(policy_logits.device)
        
        mask[:, valid_moves] = policy_logits[:, valid_moves]
        
        probs = F.softmax(mask, dim=1)
        
        return probs.squeeze() 


    def backpropagate(self, node, score):
        while node is not None:
            node.visits += 1
            node.score += score
            score = -score
            node = node.parent


            dict构建平均0.0025s × 4个叶节点 × 125次循环 = 1.25s，占search总时间2.4s的一半以上。

根本原因
dict(zip(...tolist()...)) 看起来简单，但 .tolist() 在 Windows+PyTorch 上有额外开销，加上 Python 字典的内存分配，每次构建30个键值对就要2-3ms。
解决方案：完全抛弃 p_dict 字典，改用两个 numpy 数组存储。


修改 TreeNode
pythonclass TreeNode():
    def __init__(self, parent=None):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.score = 0
        self.p_value = 0
        self.is_terminal = False
        self.is_visited = False
        # 用数组替代字典
        self.actions = None   # np.array, shape (30,)
        self.probs = None     # np.array, shape (30,)

修改赋值部分
python# 循环外批量处理，一次CPU转移
all_top_probs, all_top_actions = torch.topk(probs, 30, dim=1)
all_top_actions_np = all_top_actions.cpu().numpy()  # (batch, 30)
all_top_probs_np = all_top_probs.cpu().numpy()      # (batch, 30)

for i, (leaf, path) in enumerate(zip(leaf_nodes, paths)):
    leaf.actions = all_top_actions_np[i]  # ✅ 直接赋numpy数组，无构建开销
    leaf.probs = all_top_probs_np[i]
    leaf.is_visited = True
    self.backpropagate(path, v[i].item())

修改 get_best_child
pythondef get_best_child(self, node, c_puct=5):
    actions = node.actions  # 直接用数组
    ps = node.probs

    vs = np.array([node.children[a].visits if a in node.children else 0
                   for a in actions], dtype=np.float32)
    ss = np.array([node.children[a].score if a in node.children else 0.0
                   for a in actions], dtype=np.float32)

    q = ss / (vs + 1e-8)
    u = c_puct * ps * (np.sqrt(node.visits + 1e-8) / (1.0 + vs))

    best_idx = np.argmax(q + u)
    best_action = actions[best_idx]

    if best_action not in node.children:
        node.children[best_action] = TreeNode(parent=node)
        node.children[best_action].p_value = ps[best_idx]

    return best_action, node.children[best_action]

修改 get_policy
pythondef get_policy(self, node, board_size):
    p = np.zeros(board_size * board_size, dtype=np.float32)
    for action, child in node.children.items():
        p[action] = child.visits
    total = p.sum()
    if total > 0:
        p /= total
    return p


#select the best child of the node with the highest ucb value
    # def get_best_child(self, node, exploration_factor=5):
    #     #node = TreeNode(parent=node, state=node.state)
    #     #define best score as negative infinity

    #     best_score = -10000
    #     best_child = -1
    #     c_puct = exploration_factor
    #     for action,probs in node.p_dict.items():
    #         #check current player
    #         #todo
    #         # if child.state.player == 1: current_player = 1
    #         # else: current_player = -1
    #         if action not in node.children:
    #             child_visits = 0
    #             child_score = 0
    #         else:
    #             child = node.children[action]   
    #             child_visits = node.children[action].visits
    #             child_score = node.children[action].score
    #             child_p_value = node.children[action].p_value
            
    #         q_value = child_score / child_visits if child_visits > 0 else 0

    #         #u_value
    #         u_value = c_puct * probs * (math.sqrt(node.visits) / (1 + child_visits))
    #         score = q_value + u_value
    #         #update best child
    #         if score > best_score:
    #             best_score = score
    #             best_action = action
            
            
    #     if best_action not in node.children:
    #             node.children.update({best_action:TreeNode(parent=node)})
    #             node.children[best_action].p_value = node.p_dict[best_action]

    #     return best_action, node.children[best_action]

    array([[ 0,  0,  0,  1,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0, -1,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
       [-1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
      dtype=int8)

    def optimize(self, memory, batch_size=64):
        transitions = memory.sample(batch_size)
        #todo :four channel state
        states, actions, values = zip(*transitions)

        device = next(self.network.parameters()).device
    
        batch_states = torch.FloatTensor(np.array(states)).to(device)
        batch_actions = torch.FloatTensor(np.array(actions)).to(device)
        batch_values = torch.FloatTensor(np.array(values)).to(device).view(-1, 1)

        
        p_logits, v = self.network(batch_states)

        print(f"v mean: {v.mean().item():.3f}, std: {v.std().item():.3f}")
        print(f"target mean: {batch_values.mean().item():.3f}, std: {batch_values.std().item():.3f}")
        value_loss = F.mse_loss(v, batch_values)
        #print(batch_values.mean(), batch_values.std())
        # Policy Loss: Cross Entropy
        # 注意：p_logits 是原始输出，F.log_softmax 之后计算互熵
        log_p = F.log_softmax(p_logits, dim=1)
        policy_loss = -torch.mean(torch.sum(batch_actions * log_p, dim=1))
        
        total_loss = value_loss + policy_loss
        print(f"value_loss: {value_loss.item():.3f} | policy_loss: {policy_loss.item():.3f}")
        # 5. 反向传播与优化
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()




# LOG_FILE = "memory_debug.log"

                    # with open(LOG_FILE, "a", encoding="utf-8") as f:

                    #     f.write("\n" + "=" * 100 + "\n")
                    #     f.write(f"NEW GAME | winner = {winner}\n")
                    #     f.write("=" * 100 + "\n")

                    #     for step, (s, p, player, last_move) in enumerate(game_history):

                    #         value = 1 if player == winner else -1

                    #         # ch_state shape: (4, board_size, board_size)
                    #         ch_state = env.get_channel_state(s, last_move, player)

                    #         max_prob = np.max(p)
                    #         nonzero = np.count_nonzero(p)

                    #         board_size = int(np.sqrt(len(p)))
                    #         policy_board = np.array(p).reshape(board_size, board_size)

                    #         f.write(f"\n[STEP {step}]\n")
                    #         f.write(f"Player    : {player}\n")
                    #         f.write(f"Value     : {value}\n")
                    #         f.write(f"Last Move : {last_move}\n")
                    #         f.write(f"MaxProb   : {max_prob:.6f}\n")
                    #         f.write(f"NonZero   : {nonzero}\n")

                    #         # 原始棋盘
                    #         f.write("\n[Original Board]\n")
                    #         f.write(np.array2string(
                    #             s,
                    #             separator=' ',
                    #             max_line_width=200
                    #         ))
                    #         f.write("\n")

                    #         # policy
                    #         f.write("\n[Policy]\n")
                    #         f.write(np.array2string(
                    #             policy_board,
                    #             formatter={'float_kind': lambda x: f"{x:0.3f}"},
                    #             max_line_width=200
                    #         ))
                    #         f.write("\n")

                    #         # 输出四个 channel
                    #         f.write("\n[Channel State]\n")

                    #         for c in range(ch_state.shape[0]):

                    #             f.write(f"\n(Channel {c})\n")

                    #             f.write(np.array2string(
                    #                 ch_state[c],
                    #                 separator=' ',
                    #                 formatter={'float_kind': lambda x: f"{x:0.1f}"},
                    #                 max_line_width=200
                    #             ))

                    #             f.write("\n")

                    #         f.write("-" * 100 + "\n")

                    #         memory.append((ch_state, p, value))