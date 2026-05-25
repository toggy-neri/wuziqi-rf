"""
五子棋「即将连五」局面生成器 v5
============================================
变更：
  - next_move  → next_moves: List[int]，所有制胜格（落下任意一个即成五）
  - 验证：每个制胜格落下后均恰好产生1个连五

输出格式：
  {
      'board': np.ndarray (15,15) int8,  1=黑, -1=白, 0=空
      'value': float,                    1.0 or -1.0
      'next_moves': List[int],           所有制胜格 row*15+col
      'current_player': int,             1 or -1
  }
"""

import numpy as np
import random
from typing import List, Optional

SIZE = 15
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


# ──────────────────────────────────────────────
# 预计算窗口索引
# ──────────────────────────────────────────────

def _build_windows() -> np.ndarray:
    wins = []
    for dr, dc in DIRECTIONS:
        for r in range(SIZE):
            for c in range(SIZE):
                er, ec = r + dr * 4, c + dc * 4
                if 0 <= er < SIZE and 0 <= ec < SIZE:
                    wins.append([
                        (r + dr * i) * SIZE + (c + dc * i)
                        for i in range(5)
                    ])
    return np.array(wins, dtype=np.int32)


WINDOWS = _build_windows()   # (N_WINDOWS, 5)
N_WINDOWS = len(WINDOWS)

# 每个方向的窗口数，用于 count_fives_simple 分块
_DIR_SIZES = []
for _dr, _dc in DIRECTIONS:
    _DIR_SIZES.append(sum(
        1 for r in range(SIZE) for c in range(SIZE)
        if 0 <= r+_dr*4 < SIZE and 0 <= c+_dc*4 < SIZE
    ))
_DIR_OFFSETS = np.cumsum([0] + _DIR_SIZES)


# ──────────────────────────────────────────────
# 向量化核心
# ──────────────────────────────────────────────

def _wv(flat: np.ndarray) -> np.ndarray:
    return flat[WINDOWS]   # (N_WINDOWS, 5)


def count_fives_simple(flat: np.ndarray) -> int:
    """极大连续段>=5 → 计1，向量化版。"""
    wv = _wv(flat)
    count = 0
    for i, ds in enumerate(_DIR_SIZES):
        block = wv[_DIR_OFFSETS[i]:_DIR_OFFSETS[i+1]]   # (ds, 5)
        for player in (1, -1):
            hit = np.all(block == player, axis=1)
            if not hit.any():
                continue
            h = hit.astype(np.int8)
            starts = np.where(np.diff(np.concatenate([[0], h])) == 1)[0]
            count += len(starts)
    return count


def find_winning_moves_fast(flat: np.ndarray, player: int) -> np.ndarray:
    """返回所有制胜格线性索引（已去重）。"""
    wv = _wv(flat)
    valid = (np.sum(wv == player, axis=1) == 4) & (np.sum(wv == 0, axis=1) == 1)
    if not valid.any():
        return np.array([], dtype=np.int32)
    vw = WINDOWS[valid]                         # (k, 5)
    vwv = wv[valid]                             # (k, 5)
    empty_col = np.argmax(vwv == 0, axis=1)     # (k,)
    winning = vw[np.arange(len(vw)), empty_col] # (k,)
    return np.unique(winning)


# ──────────────────────────────────────────────
# 主生成函数
# ──────────────────────────────────────────────

def generate_single_sample(rng: random.Random) -> Optional[dict]:
    board = np.zeros(SIZE * SIZE, dtype=np.int8)

    current_player = rng.choice([1, -1])
    opponent = -current_player

    # 1. 随机选5格窗口，放4子
    win_idx   = rng.randrange(N_WINDOWS)
    window    = WINDOWS[win_idx]
    empty_pos = rng.randrange(5)
    seed_move = int(window[empty_pos])
    fill_cells = window[np.arange(5) != empty_pos]

    board[fill_cells] = current_player

    if count_fives_simple(board) > 0:
        return None
    if seed_move not in find_winning_moves_fast(board, current_player):
        return None

    # 2. 子数目标
    # ──────────────────────────────────────────────────────────
    # 核心原则：黑白子数与胜负完全解耦，模型无法通过数子数推断胜负。
    #
    # 做法：独立随机决定黑子数和白子数，不绑定"谁落子=谁赢"。
    #   n_current ∈ [base, base+range]  (current_player的子数，含已有4颗)
    #   n_opponent ∈ [base, base+range]  独立采样
    # 这样子数差 在 黑胜/白胜 两组中分布完全相同，相关性为零。
    #
    # 代价：不再严格符合先手奇偶约束，但对于战术训练数据这是可接受的，
    #       模型需要从棋盘内容（4连的位置）而非元信息判断胜负。
    # ──────────────────────────────────────────────────────────
    base = rng.randint(0, 60)
    n_current  = base + rng.randint(0, 4)   # current_player 总子数（含已有4颗）
    n_opponent = base + rng.randint(0, 4)   # opponent 总子数，独立采样

    n_extra_current  = n_current - 4
    n_extra_opponent = n_opponent

    if n_extra_current < 0 or n_extra_opponent < 0:
        return None
    total_needed = n_extra_current + n_extra_opponent

    # 3. 候选空格（排除所有当前制胜格，避免扰动误占）
    current_winning = find_winning_moves_fast(board, current_player)
    forbidden = set(current_winning.tolist())

    empty_mask = (board == 0)
    for f in forbidden:
        empty_mask[f] = False
    candidates = np.where(empty_mask)[0]

    if len(candidates) < total_needed:
        return None

    colors = np.array(
        [current_player] * n_extra_current + [opponent] * n_extra_opponent,
        dtype=np.int8
    )

    # 4. 批量随机排列重试
    board_backup = board.copy()
    MAX_PERM = 40
    success = False

    for _ in range(MAX_PERM):
        chosen = candidates.copy()
        np.random.shuffle(chosen)
        chosen = chosen[:total_needed]
        np.random.shuffle(colors)

        board[chosen] = colors

        # 验证
        if count_fives_simple(board) > 0:
            board[:] = board_backup
            continue

        wm_cur = np.unique(find_winning_moves_fast(board, current_player))
        # 必须恰好1个制胜格，且就是我们选定的next_move
        if len(wm_cur) != 1 or wm_cur[0] != seed_move:
            board[:] = board_backup
            continue

        wm_opp = find_winning_moves_fast(board, opponent)
        if len(wm_opp) > 0:
            board[:] = board_backup
            continue

        success = True
        break

    if not success:
        return None

    # 5. 最终验证
    if count_fives_simple(board) != 0:
        return None

    # 收集所有制胜格
    all_winning = find_winning_moves_fast(board, current_player)
    if len(all_winning) == 0:
        return None

    # 验证每个制胜格落下后恰好产生1个连五
    valid_moves = []
    for mv in np.unique(all_winning).tolist():
        board[mv] = current_player
        fives = count_fives_simple(board)
        board[mv] = 0
        if fives == 1:
            valid_moves.append(mv)

    # 关键过滤：只保留恰好1个制胜格的局面
    # 多制胜格会导致训练标签歧义（模型预测另一个合法格也算错）
    if len(valid_moves) != 1:
        return None

    next_move = valid_moves[0]

    # 子数验证
    black = int(np.sum(board == 1))
    white = int(np.sum(board == -1))
    if current_player == 1:
        if black != white: return None
    else:
        if black != white + 1: return None

    return {
        'board':          board.reshape(SIZE, SIZE).copy(),
        'value':          float(current_player),
        'next_move':      next_move,   # int，唯一制胜格
        'current_player': current_player,
    }


# ──────────────────────────────────────────────
# 批量生成
# ──────────────────────────────────────────────

def generate_samples(n: int = 100, seed: Optional[int] = None, verbose: bool = False) -> List[dict]:
    if seed is not None:
        np.random.seed(seed)
    rng = random.Random(seed)
    samples, attempts = [], 0

    while len(samples) < n and attempts < n * 500:
        attempts += 1
        s = generate_single_sample(rng)
        if s is not None:
            samples.append(s)
            if verbose and len(samples) % 200 == 0:
                print(f"  [{len(samples)}/{n}] 尝试={attempts} 成功率={len(samples)/attempts*100:.1f}%")

    if verbose:
        print(f"完成：{len(samples)} 样本，总尝试 {attempts}，成功率 {len(samples)/attempts*100:.1f}%")
    return samples


# ──────────────────────────────────────────────
# 工具：可视化
# ──────────────────────────────────────────────

def print_board(board: np.ndarray, next_move: int):
    symbols = {0: '·', 1: '●', -1: '○'}
    nr, nc = next_move // SIZE, next_move % SIZE
    print("   " + " ".join(f"{c:2}" for c in range(SIZE)))
    for r in range(SIZE):
        row = f"{r:2} "
        for c in range(SIZE):
            row += " ★" if (r == nr and c == nc) else f" {symbols[board[r, c]]}"
        print(row)
    print()


# ──────────────────────────────────────────────
# 演示 & 验证
# ──────────────────────────────────────────────

if __name__ == '__main__':
    import time
    from collections import Counter

    print("=" * 60)
    print("五子棋「即将连五」生成器 v5  唯一制胜格版")
    print("  只保留恰好1个制胜格的局面，训练标签无歧义")
    print("  ★ = next_move（唯一制胜格）")
    print("=" * 60)

    t0 = time.time()
    samples = generate_samples(n=10000, seed=None, verbose=True)
    elapsed = time.time() - t0
    print(f"\n耗时: {elapsed:.2f}s，速度: {len(samples)/elapsed:.0f} samples/s")

    # 全量验证
    errors = 0
    for s in samples:
        flat = s['board'].ravel()
        cp   = s['current_player']
        mv   = s['next_move']

        # 落子前无连五
        if count_fives_simple(flat) != 0:
            errors += 1; continue

        # 落子后恰好1个连五
        f2 = flat.copy(); f2[mv] = cp
        if count_fives_simple(f2) != 1:
            errors += 1; continue

        # 确认是唯一制胜格
        wm = np.unique(find_winning_moves_fast(flat, cp))
        if len(wm) != 1 or wm[0] != mv:
            errors += 1; continue

        # 子数平衡
        b, w = int(np.sum(flat == 1)), int(np.sum(flat == -1))
        if cp == 1 and b != w:       errors += 1
        elif cp == -1 and b != w+1:  errors += 1

    print(f"验证错误: {errors}/{len(samples)}  {'✓ 全部通过' if errors==0 else '✗ 有错误'}")

    # 展示前3个
    for i, s in enumerate(samples[:3]):
        b = int(np.sum(s['board'] == 1))
        w = int(np.sum(s['board'] == -1))
        cp = '黑(1)' if s['current_player'] == 1 else '白(-1)'
        mv = s['next_move']
        print(f"\n─── 样本 #{i+1} ───  current={cp}  黑={b} 白={w}  制胜格=({mv//15},{mv%15})")
        print_board(s['board'], mv)
    import pickle # 保存 
    with open("basic_skill4.pkl", "wb") as f: 
        pickle.dump(samples, f) 
        print("已保存到 basic_skill4.pkl")