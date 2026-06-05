"""
五子棋 Minimax + Alpha-Beta 剪枝  v2
======================================
优化：
  - 候选走法限制上限 MAX_CANDIDATES=20，按启发分截断
  - 评估函数完全向量化（numpy窗口批量计算），去掉Python循环
  - 走法排序：先算每个候选的浅层分，高分优先搜索（剪枝效率显著提升）
  - 立即胜负检测放在搜索最前面，命中即返回无需评估

深度3目标：< 1s / 步
"""

import pickle
import os
import numpy as np
from typing import Optional, Tuple, List
import time

SIZE = 15
INF  = 10_000_000
MAX_CANDIDATES = 20   # 每层最多考虑的候选走法数

# ─────────────────────────────────────────────────────────
# 预计算：所有5格窗口
# ─────────────────────────────────────────────────────────
DIRECTIONS = [(0,1),(1,0),(1,1),(1,-1)]

def _build_windows():
    wins = []
    for dr, dc in DIRECTIONS:
        for r in range(SIZE):
            for c in range(SIZE):
                er, ec = r+dr*4, c+dc*4
                if 0 <= er < SIZE and 0 <= ec < SIZE:
                    wins.append([(r+dr*i)*SIZE+(c+dc*i) for i in range(5)])
    return np.array(wins, dtype=np.int32)

WINDOWS = _build_windows()   # (N_WIN, 5)
N_WIN   = len(WINDOWS)



# ─────────────────────────────────────────────────────────
# 胜负判断（向量化）
# ─────────────────────────────────────────────────────────

def check_win(flat: np.ndarray, player: int) -> bool:
    wv = flat[WINDOWS][:, :5]  # 只看前5格即可判定连五
    return bool(np.any(np.all(wv == player, axis=1)))



# ─────────────────────────────────────────────────────────
# 评估函数（完全向量化）
# ─────────────────────────────────────────────────────────
#
# 对每个5格窗口统计 (player棋子数, opponent棋子数)
# 用查表得分，批量sum。
#
# 分值表：key=(player棋子数, opponent棋子数)
# opponent==0 才有意义（窗口未被对手阻断）

# (p_cnt, o_cnt) -> score
# 活四/活三的分数大幅提升！活三=冲四，都是致命威胁！
_SCORE_TABLE = {
    # (p_cnt, o_cnt) -> score for player
    (5, 0): 100_000,   # 连五
    (4, 0):  20_000,   # 活四/冲四（这里不区分，由搜索深度弥补）
    (4, 1):  18_000,    # 活四/冲四（这里不区分，由搜索深度弥补）
    (3, 0):   4_100,    # 活三/眠三
    (2, 0):     275,    # 活二/眠二
    (1, 0):      10,    # 单子
}
# 预构建向量化查表：ndarray indexed by (p_cnt, o_cnt)
_SCORE_ARR = np.zeros((6, 6), dtype=np.int32)

for (p, o), v in _SCORE_TABLE.items():
    _SCORE_ARR[p, o] = v


def evaluate_fast(flat: np.ndarray, ai_player: int) -> int:
    opp = -ai_player
    wv  = flat[WINDOWS]

    ai_cnt  = np.sum(wv == ai_player, axis=1).astype(np.int32)   
    opp_cnt = np.sum(wv == opp,       axis=1).astype(np.int32)   

    # AI 得分
    ai_mask  = (opp_cnt == 0)
    ai_score = int(np.sum(_SCORE_ARR[ai_cnt[ai_mask], opp_cnt[ai_mask]]))
    ai_score += evaluate_threats(flat, ai_player)  # 加入进攻雷达

    # 对手得分
    hm_mask  = (ai_cnt == 0)
    hm_score = int(np.sum(_SCORE_ARR[opp_cnt[hm_mask], ai_cnt[hm_mask]]))
    hm_score += evaluate_threats(flat, opp)        # 加入防守雷达（极其重要！）

    return ai_score - hm_score


# 候选走法的启发分也要同步加强！
def get_candidates_fast(board: np.ndarray,
                        flat: np.ndarray,
                        ai_player: int,
                        radius: int = 2,
                        max_cands: int = MAX_CANDIDATES) -> List[int]:
    if not np.any(board != 0):
        return [SIZE//2 * SIZE + SIZE//2]

    occ_r, occ_c = np.where(board != 0)
    cand_set = set()
    for r, c in zip(occ_r, occ_c):
        r0, r1 = max(0, r-radius), min(SIZE, r+radius+1)
        c0, c1 = max(0, c-radius), min(SIZE, c+radius+1)
        for nr in range(r0, r1):
            for nc in range(c0, c1):
                if board[nr, nc] == 0:
                    cand_set.add(nr * SIZE + nc)

    if not cand_set:
        return []

    cands  = list(cand_set)
    opp    = -ai_player
    scores = []

    for lin in cands:
        win_idx = _CELL_WINDOWS[lin]
        if len(win_idx) == 0:
            scores.append(0)
            continue
        wv_sub   = flat[WINDOWS[win_idx]]
        ai_cnt   = np.sum(wv_sub == ai_player, axis=1)
        opp_cnt  = np.sum(wv_sub == opp,       axis=1)
        
        # 进攻静态分 + 进攻雷达分
        s_atk = int(np.sum(_SCORE_ARR[np.minimum(ai_cnt+1, 5), opp_cnt]))
        flat[lin] = ai_player
        s_atk += np.sum(np.all(flat[WINDOWS] == ai_player, axis=1)) * 20000
        flat[lin] = 0
        
        # 防守静态分 + 防守雷达分
        s_def = int(np.sum(_SCORE_ARR[np.minimum(opp_cnt+1, 5), ai_cnt]))
        flat[lin] = opp
        s_def += np.sum(np.all(flat[WINDOWS] == opp, axis=1)) * 20000
        flat[lin] = 0
        
        scores.append(max(s_atk, s_def))
    scores_np = np.array(scores, dtype=np.float32)    
    max_score = np.max(scores_np)
    min_score = np.min(scores_np)
    score_range = max(max_score - min_score, 1.0) # 防止除0
    
    # 生成 [-0.5, 0.5] 的随机噪声，并缩放到不影响大局的微小尺度 (1%)
    noise = (np.random.rand(len(scores_np)) - 0.5) * score_range * 0.01
    
    # 将噪声叠加到分数上
    perturbed_scores = scores_np + noise
    order = np.argsort(-perturbed_scores)[:max_cands]
    return [cands[i] for i in order]



# ─────────────────────────────────────────────────────────
# 预计算：每个格子参与的窗口掩码（启动时一次性完成）
# ─────────────────────────────────────────────────────────

# 重跑构建
_CELL_WINDOWS: List[np.ndarray] = []

def _build_cell_windows():
    global _CELL_WINDOWS
    _CELL_WINDOWS = []
    for lin in range(SIZE * SIZE):
        mask = np.where(np.any(WINDOWS == lin, axis=1))[0]
        _CELL_WINDOWS.append(mask)

_build_cell_windows()

def evaluate_threats(flat: np.ndarray, player: int) -> int:
    """
    向量化二次评估：检测一步致胜的威胁（冲四/活四）。
    如果一个空位落子后能形成新的连五，说明该空位是致命威胁点。
    """
    wv = flat[WINDOWS]  # (N_WIN, 5)
    p_cnt = np.sum(wv == player, axis=1)
    e_cnt = np.sum(wv == 0, axis=1)
    o_cnt = np.sum(wv == -player, axis=1)
    
    # 1. 找到差一子连五的窗口 (4子1空0对手)
    valid = (p_cnt == 4) & (e_cnt == 1) & (o_cnt == 0)
    if not valid.any():
        return 0
        
    vw = WINDOWS[valid]
    vwv = wv[valid]
    # 找到这些窗口中空位的线性索引
    ei = np.argmax(vwv == 0, axis=1)
    empties = vw[np.arange(len(vw)), ei]
    
    # 2. 假想落子：在空位处放置 player
    flat[empties] = player
    
    # 3. 二次检测：落子后，产生了多少个连五？
    wv2 = flat[WINDOWS]
    new_fives = np.sum(np.all(wv2 == player, axis=1))
    
    # 4. 撤销落子
    flat[empties] = 0
    
    # 每个一步致胜点给与极高分数，双杀(冲四活四等)分数翻倍
    return int(new_fives) * 20000


def get_candidates_fast(board: np.ndarray,
                        flat: np.ndarray,
                        ai_player: int,
                        radius: int = 2,
                        max_cands: int = MAX_CANDIDATES) -> List[int]:
    if not np.any(board != 0):
        return [SIZE//2 * SIZE + SIZE//2]

    occ_r, occ_c = np.where(board != 0)
    cand_set = set()
    for r, c in zip(occ_r, occ_c):
        r0, r1 = max(0, r-radius), min(SIZE, r+radius+1)
        c0, c1 = max(0, c-radius), min(SIZE, c+radius+1)
        for nr in range(r0, r1):
            for nc in range(c0, c1):
                if board[nr, nc] == 0:
                    cand_set.add(nr * SIZE + nc)

    if not cand_set:
        return []

    cands  = list(cand_set)
    opp    = -ai_player
    scores = []

    for lin in cands:
        win_idx = _CELL_WINDOWS[lin]
        if len(win_idx) == 0:
            scores.append(0)
            continue
        wv_sub   = flat[WINDOWS[win_idx]]                 # (k, 6)
        ai_cnt   = np.sum(wv_sub == ai_player, axis=1)
        opp_cnt  = np.sum(wv_sub == opp,       axis=1)
        
        # 进攻分 (模拟落子 ai_player，相当于原来6格里的空位变成了ai)
        s_atk = int(np.sum(_SCORE_ARR[np.minimum(ai_cnt+1, 5), opp_cnt]))
        
        # 防守分 (模拟落子 opp)
        s_def = int(np.sum(_SCORE_ARR[np.minimum(opp_cnt+1, 5), ai_cnt]))
        
        scores.append(max(s_atk, s_def))

    order  = np.argsort(-np.array(scores))[:max_cands]
    return [cands[i] for i in order]


# ─────────────────────────────────────────────────────────
# 立即胜负检测（向量化，不逐格试放）
# ─────────────────────────────────────────────────────────

def find_winning_moves_vec(flat: np.ndarray, player: int) -> np.ndarray:
    opp = -player
    wv  = flat[WINDOWS][:, :5]  # 只看前5格
    p_cnt = np.sum(wv == player, axis=1)
    e_cnt = np.sum(wv == 0,      axis=1)
    o_cnt = np.sum(wv == opp,    axis=1)
    valid = (p_cnt == 4) & (e_cnt == 1) & (o_cnt == 0)
    if not valid.any():
        return np.array([], dtype=np.int32)
    vw  = WINDOWS[valid][:, :5]
    vwv = wv[valid]
    ei  = np.argmax(vwv == 0, axis=1)
    return np.unique(vw[np.arange(len(vw)), ei])


# ─────────────────────────────────────────────────────────
# Minimax + Alpha-Beta
# ─────────────────────────────────────────────────────────

def minimax(flat: np.ndarray,
            board: np.ndarray,
            depth: int,
            alpha: int,
            beta: int,
            is_maximizing: bool,
            ai_player: int) -> int:

    human = -ai_player

    # 终局检测（向量化，比逐格快）
    if check_win(flat, ai_player): return  INF + depth
    if check_win(flat, human):     return -INF - depth
    if depth == 0:                 return evaluate_fast(flat, ai_player)

    cands = get_candidates_fast(board, flat, ai_player if is_maximizing else human)
    if not cands:
        return evaluate_fast(flat, ai_player)

    player = ai_player if is_maximizing else human

    if is_maximizing:
        best = -INF
        for lin in cands:
            flat[lin] = player
            board[lin // SIZE, lin % SIZE] = player
            val = minimax(flat, board, depth-1, alpha, beta, False, ai_player)
            flat[lin] = 0
            board[lin // SIZE, lin % SIZE] = 0
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best
    else:
        best = INF
        for lin in cands:
            flat[lin] = player
            board[lin // SIZE, lin % SIZE] = player
            val = minimax(flat, board, depth-1, alpha, beta, True, ai_player)
            flat[lin] = 0
            board[lin // SIZE, lin % SIZE] = 0
            if val < best:
                best = val
            if best < beta:
                beta = best
            if alpha >= beta:
                break
        return best


# ─────────────────────────────────────────────────────────
# 主接口
# ─────────────────────────────────────────────────────────

def get_best_move(board: np.ndarray,
                  ai_player: int,
                  depth: int = 2,
                  return_policy: bool = False) -> Optional[Tuple[int, int]]:
    """
    返回 AI 最佳落子 (row, col)。
    return_policy: 如果为 True，则额外返回 shape=(225,) 的策略概率矩阵
    """
    human = -ai_player
    flat  = board.ravel().copy()
    
    # 默认空策略
    policy_matrix = np.zeros(SIZE * SIZE, dtype=np.float32)

    if not np.any(board != 0):
        r = np.random.randint(2, SIZE-2)
        c = np.random.randint(2, SIZE-2)
        policy_matrix[r * SIZE + c] = 1.0
        if return_policy: return (r,c), policy_matrix
        return (SIZE//2, SIZE//2)

    # 1. 检查立即制胜
    wm = find_winning_moves_vec(flat, ai_player)
    if len(wm) > 0:
        best_lin = int(wm[0])
        # 制胜点是 100% 概率
        policy_matrix[wm] = 1.0 / len(wm) 
        r, c = best_lin // SIZE, best_lin % SIZE
        if return_policy: return (r, c), policy_matrix
        return (r, c)

    # 2. 检查必须封堵
    wm = find_winning_moves_vec(flat, human)
    if len(wm) > 0:
        best_lin = int(wm[0])
        # 封堵点也是 100% 必走
        policy_matrix[wm] = 1.0 / len(wm)
        r, c = best_lin // SIZE, best_lin % SIZE
        if return_policy: return (r, c), policy_matrix
        return (r, c)

    # 3. Minimax 搜索
    cands = get_candidates_fast(board, flat, ai_player)
    if not cands:
        if return_policy: return None, policy_matrix
        return None

    cand_scores = []
    best_score, best_lin = -INF, cands[0]
    alpha = -INF

    for lin in cands:
        flat[lin] = ai_player; board[lin//SIZE, lin%SIZE] = ai_player
        # 因为对于对手来说是最小化，所以传入的 beta 是 INF，alpha 是当前最大值
        score = minimax(flat, board, depth-1, alpha, INF, False, ai_player)
        flat[lin] = 0; board[lin//SIZE, lin%SIZE] = 0
        
        cand_scores.append(score)
        if score > best_score:
            best_score, best_lin = score, lin
        alpha = max(alpha, best_score)

    r, c = best_lin // SIZE, best_lin % SIZE
    print(f"[Minimax] 落子 ({r},{c})  分值={best_score}  深度={depth}")
    # ================= 核心新增：生成策略矩阵 =================
    if return_policy:
        # 将得分转为 numpy 数组
        scores_np = np.array(cand_scores, dtype=np.float32)
        
        # 温度系数 T：T 越大，分布越平滑（探索越多）；T 越小，分布越尖锐（利用越多）
        # 对于深度2的 Minimax，分值通常在几千到几万，T=100 是一个不错的起点
        temperature = 100.0 
        
        # 缩放得分并减去最大值，防止 Softmax 溢出
        scaled_scores = (scores_np - np.max(scores_np)) / temperature
        
        # Softmax 计算概率
        exp_scores = np.exp(scaled_scores)
        probabilities = exp_scores / np.sum(exp_scores)
        
        # 将概率填入策略矩阵对应位置
        for i, lin in enumerate(cands):
            policy_matrix[lin] = probabilities[i]
            
        return (r, c), policy_matrix
    # ==========================================================


    return (r, c)



# ─────────────────────────────────────────────────────────
# 对弈封装
# ─────────────────────────────────────────────────────────

class Minimax:
    """
    对弈逻辑封装：
      minimax_vs_minimax()  两个 minimax 互弈（验证用）
      ai_vs_minimax()       你的 AI 模型 vs minimax
    """
    def __init__(self, board_size: int = 15):
        self.board   = np.zeros((board_size, board_size), dtype=np.int8)
        self.current = 1

    def reset(self):
        self.board[:] = 0
        self.current  = 1

    def is_over(self):
        flat = self.board.ravel()
        if check_win(flat,  1): return  1
        if check_win(flat, -1): return -1
        if not np.any(self.board == 0): return 0
        return None

    def print_board(self, last=None):
        sym = {0:'·', 1:'●', -1:'○'}
        print("   " + " ".join(f"{c:2}" for c in range(SIZE)))
        for r in range(SIZE):
            row = f"{r:2} "
            for c in range(SIZE):
                row += " ★" if last==(r,c) else f" {sym[self.board[r,c]]}"
            print(row)
        print()

    def play_move(self, r, c):
        self.board[r, c] = self.current
        self.current = -self.current
        return self.is_over()

    def minimax_vs_minimax(self, depth_black=3, depth_white=3, verbose=True, max_steps=60):
        self.reset()
        depths = {1: depth_black, -1: depth_white}
        last   = None
        for step in range(max_steps):
            if self.is_over() is not None: break
            p    = self.current
            move = get_best_move(self.board, p, depth=depths[p])
            if not move: break
            last = move
            self.play_move(*move)
            if verbose:
                self.print_board(last)
            if self.is_over() is not None: break
        result = self.is_over()
        if verbose:
            print({1:"黑胜", -1:"白胜", 0:"平局", None:"未结束"}.get(result))
        return result
    def get_minimax_action(self, board: np.ndarray, current_player: int, depth: int = 1) -> int:
        """
        根据当前局面返回 minimax 的最佳动作（线性索引 row*SIZE+col）。
        
        board          : (15,15) int8，直接传入 env 的棋盘
        current_player : 当前走子方（minimax 执此方）
        返回           : int，action = row*SIZE + col
        """
        move, policy_matrix = get_best_move(board, current_player, depth=depth, return_policy=True)
        if move is None:
            # 兜底：随机选一个空格
            empty = np.argwhere(board == 0)
            r, c  = empty[np.random.randint(len(empty))]
            return int(r) * SIZE + int(c)
        r, c = move
        action = r * SIZE + c
    # 验证：env解析action的方式和这里是否一致
        #rint(f"[DEBUG] move=({r},{c}) action={action} 反解=({action//SIZE},{action%SIZE})")
        return action, policy_matrix
    def ai_vs_minimax(self,
                      your_ai_func,
                      ai_player: int = 1,
                      minimax_depth: int = 1,
                      verbose: bool = True,
                      max_steps: int = 225):
        """
        你的 AI 模型 vs Minimax。

        your_ai_func : callable(board: np.ndarray, current_player: int) -> (row, col)
        ai_player    : 你的AI执哪方，1=黑 / -1=白
        """
        self.reset()
        last = None
        for step in range(max_steps):
            if self.is_over() is not None: break
            p   = self.current
            tag = "你的AI" if p == ai_player else "Minimax"
            if p == ai_player:
                move = your_ai_func(self.board.copy(), p)
            else:
                move = get_best_move(self.board, p, depth=minimax_depth)
            if not move: break
            last = move
            self.play_move(*move)
            if verbose:
                name = "黑" if p == 1 else "白"
                print(f"第{step+1}步 [{tag}] {name} → {move}")
                self.print_board(last)
            if self.is_over() is not None: break

        result = self.is_over()
        if verbose:
            labels = {
                 1: "黑胜" + (" (你的AI)" if ai_player== 1 else " (Minimax)"),
                -1: "白胜" + (" (你的AI)" if ai_player==-1 else " (Minimax)"),
                 0: "平局",
            }
            print(f"\n对弈结束：{labels.get(result, '未结束')}")
        return result
    def generate_games(self, num_games: int, save_path: str = "minimax_dataset.pkl"):
        dataset = []
        print(f"开始生成 {num_games} 局 Minimax 对弈棋谱...")
        
        for game_idx in range(num_games):
            self.reset()
            game_history = [] 
            depth_black = np.random.randint(2, 3)
            depth_white = np.random.randint(2, 3)
            
            winner = None
            for step in range(225):
                if self.is_over() is not None:
                    winner = self.is_over()
                    break
                    
                p = self.current
                depth = depth_black if p == 1 else depth_white
                
                # 调用新接口，获取走步和策略
                move, policy = get_best_move(self.board.copy(), p, depth=depth, return_policy=True)
                # print(step)
                # print(np.nonzero(policy))
                if not move:
                    winner = 0 
                    break

                # 记录落子前的状态和策略
                if self.is_over() is None:
                    game_history.append({
                        'game_id': game_idx,        # <--- 新增：标记属于哪一局
                        'board': self.board.copy(), 
                        'current_player': p,
                        'next_moves': policy.tolist() if isinstance(policy, np.ndarray) else policy,
                        'value': 0.0
                    })
                self.play_move(*move)
                

            if winner is None: winner = 0

            # 回填 Value
            for data in game_history:
                if winner == 0:
                    data['value'] = 0.0
                elif data['current_player'] == winner:
                    data['value'] = 1.0
                else:
                    data['value'] = -1.0
                    
            dataset.extend(game_history)
            
            if (game_idx + 1) % 5 == 0:
                print(f"已完成 {game_idx + 1}/{num_games} 局, 累计数据条数: {len(dataset)}")

        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"棋谱生成完毕！共 {len(dataset)} 条数据，已保存至: {os.path.abspath(save_path)}")



import pickle
import numpy as np

def check_dataset_stats(pkl_path: str):
    print(f"正在加载文件: {pkl_path} ...")
    try:
        with open(pkl_path, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    total_samples = len(dataset)
    if total_samples == 0:
        print("文件为空！")
        return

    # 检查是否包含 game_id 字段
    has_game_id = 'game_id' in dataset[0]
    if not has_game_id:
        print("⚠️ 数据中缺少 game_id 字段，无法准确统计！请重新生成棋谱。")
        return

    # 按 game_id 分组
    games = {}
    for data in dataset:
        gid = data['game_id']
        if gid not in games:
            games[gid] = []
        games[gid].append(data)

    total_games = len(games)
    black_wins = 0
    white_wins = 0
    draws = 0
    game_lengths = []
    zero_policy_count = 0

    for gid, frames in games.items():
        # 一局的步数 = 帧数（包含落子前和落子后）
        game_lengths.append(len(frames))

        # 找到这一局的最终 value
        # 按照当前的 generate_games，最后一帧的 value 就是胜败结果
        # 但是！由于我们之前没有补录终局帧，最后一帧其实是落子前的状态
        # 为了更准确，我们看最后一帧的 current_player 和 value
        last_frame = frames[-1]
        final_value = last_frame['current_player']
        
        # 统计胜负
        if final_value == 1.0:
            black_wins += 1
        elif final_value == -1.0:
            white_wins += 1
        elif final_value == 0.0:
            # 平局可能 value 是 0，也可能是不完整的对局
            # 检查棋盘是否满了
            if np.count_nonzero(last_frame['board']) == 225:
                draws += 1
            else:
                # 未下满但 value=0，可能是中途中断的游戏
                draws += 1

        # 统计无效策略
        for frame in frames:
            policy = frame['next_moves']
            if isinstance(policy, list):
                if sum(policy) == 0:
                    zero_policy_count += 1
            elif isinstance(policy, np.ndarray):
                if np.sum(policy) == 0:
                    zero_policy打分 += 1

    print("=" * 40)
    print("📊 棋谱数据统计报告")
    print("=" * 40)
    print(f"总样本数 (帧数): {total_samples}")
    print(f"总对局数: {total_games}")
    print("-" * 40)
    if total_games > 0:
        print(f"⚫ 黑方胜局: {black_wins} ({black_wins/total_games*100:.1f}%)")
        print(f"⚪ 白方胜量: {white_wins} ({white_wins/total_games*100:.1f}%)")
        print(f"🤝 平局:     {draws} ({draws/total_games*100:.1f}%")
        print(f"📏 平均每局步数: {np.mean(game_lengths):.1f} 步 (最短: {min(game_lengths)}, 最长: {max(game_lengths)})")
    
    print("-" * 40)
    if zero_policy_count > 0:
        print(f"⚠️ 警告: 发现 {zero_policy_count} 全0策略 (占比 {zero_policy_count/total_samples*100:.2f}%)！")
        print("   全0策略通常出现在终局帧，训练时建议屏蔽这些帧的 Policy Loss。")
    else:
        print("✅ 策略有效性: 良好，未发现全0策略。")
    print("=" * 40)

# 使用示例：
# check_dataset_stats("minimax_dataset.pkl")

# 使用示例：
# check_dataset_stats("minimax_dataset.pkl")

# ─────────────────────────────────────────────────────────
# 演示
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    game = Minimax()
    game.generate_games(num_games=2000, save_path="minimax_data2.pkl")
    check_dataset_stats("minimax_data2.pkl")
    # 生成 20 局，黑方深度2，白方深度2，保存到当前目录
