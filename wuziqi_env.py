import numpy as np
from typing import Tuple, Optional, List


#Named Tuple todo  
#todo:get latest move from the environment function
class WuziqiEnv:
    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)

        #active 
        self.white_plane = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        self.black_plane = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        self.last_move_plane = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        self.current_player_plane = {
            1:np.ones((self.board_size, self.board_size), dtype=np.float32),
            -1:-np.ones((self.board_size, self.board_size), dtype=np.float32)
        }
        #maintain move info
        self.move_memory = []
    
        self.done = False
        self.last_move = None
        #black first
        self.current_player = -1
        
    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.winner = None
        self.done = False
        self.last_move = None
        self.move_memory = []                                                    # ← 加这行
        self.white_plane = np.zeros((self.board_size, self.board_size), dtype=np.float32)  # ← 加这行
        self.black_plane = np.zeros((self.board_size, self.board_size), dtype=np.float32)  # ← 加这行
        self.last_move_plane = np.zeros((self.board_size, self.board_size), dtype=np.float32)  # ← 加这行
        return self.board
    def step(self, action:int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Args:
            action: (row_size * col_size + col)

        Returns:
            next_state: 下一步后的棋盘
            reward: 奖励值
            done: 游戏是否结束
            info: 额外信息
        """
        if self.done:
            return self.board.copy(), 0, True, {"winner": self.winner}  #在search中调用了check_win,当时赋值了self.done = True,但是未回滚
        if action is None:
            return self.board.copy(), 0, False, {"pass": True}
        row = action // self.board_size
        col = action % self.board_size
        
        if not self._is_valid_move(row, col):
            return self.board.copy(), -10, False, {"invalid_move": True}
        
        

        #maintain the channel state
        self.board[row, col] = self.current_player
        self.white_plane[row, col] = 1 if self.current_player == 1 else 0
        self.black_plane[row, col] = -1 if self.current_player == -1 else 0

        self.last_move_plane = np.zeros((self.board_size, self.board_size))
        self.last_move_plane[row, col] = 1
        
        self.last_move = action
        self.move_memory.append((action,self.current_player,self.last_move))
        is_win,winner = self._check_win(action)
        if is_win:
            self.done = True
            self.winner = winner
            reward = 1.0 if winner == 1 else -1.0
            return self.board.copy(), reward, True, {"winner": winner}
        
        if self._is_board_full():
            self.done = True
            self.winner = 0
            return self.board.copy(), 0, True, {"winner": 0}
        
        
        self.current_player = -self.current_player
        return self.board.copy(), 0, False, {}
        
    def _is_valid_move(self, row: int, col: int) -> bool:
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        return self.board[row, col] == 0
    
    def _check_win(self, position:int) -> Tuple[bool, int]:
        if not position:
            return False,None
        
        row = position // self.board_size
        col = position % self.board_size
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                winner = self.current_player
                return True,winner

        return False,None
    def undo(self):
        if not self.move_memory:  
            return
        try:
            action,player,position = self.move_memory.pop()
        except IndexError as e:
            print("UNDO ERROR:", e)
            return
        row = position // self.board_size
        col = position % self.board_size
        self.board[row, col] = 0
        self.white_plane[row, col] = 0
        self.black_plane[row, col] = 0
        if self.move_memory:
            self.last_move = self.move_memory[-1][2]

        self.winner = None
        self.current_player = -self.current_player
        self.done = False
    
    def _is_board_full(self) -> bool:
        return np.all(self.board != 0)
    
    def get_valid_moves(self,board) -> List[int]:
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0:
                    moves.append(15 * i + j)
        return moves
    
    def get_state(self) -> np.ndarray:
        return self.board.copy()


    #HERE： create the channel state too frequently,be abandoned
    # def get_channel_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     black_state = (state == 1).astype(np.float32)
    #     white_state = (state == -1).astype(np.float32)
    #     last_move_state = np.zeros((self.board_size, self.board_size), dtype=np.float32)
    #     if self.last_move is not None:
    #         row = self.last_move % self.board_size
    #         col = self.last_move // self.board_size
    #         last_move_state[row, col] = 1.0
    #     current_player_state = np.full((self.board_size, self.board_size), self.current_player, dtype=np.float32)
    #     return black_state, white_state, last_move_state, current_player_state
    def get_channel_state(self, board: np.ndarray, last_move: int,current_player: int) -> np.ndarray:
        black_plane = (board == 1).astype(np.float32)
        white_plane = (board == -1).astype(np.float32)
        last_move_plane = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        if last_move is not None:
            row = last_move // self.board_size
            col = last_move % self.board_size
            last_move_plane[row, col] = 1.0
        return np.array([
            black_plane,
            white_plane,
            last_move_plane,
            self.current_player_plane[current_player]
        ], dtype=np.float32)
    

    def render(self) -> str:
        symbols = {0: '.', 1: 'X', -1: 'O'}
        board_str = "  " + " ".join(f"{i:2d}" for i in range(self.board_size)) + "\n"
        for i in range(self.board_size):
            row_str = f"{i:2d} "
            for j in range(self.board_size):
                row_str += symbols[self.board[i, j]] + "  "
            board_str += row_str + "\n"
        return board_str
    
    def copy(self) -> 'WuziqiEnv':
        new_env = WuziqiEnv(self.board_size)
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.winner = self.winner
        new_env.done = self.done
        new_env.last_move = self.last_move
        new_env.move_memory = self.move_memory.copy()
        new_env.white_plane = self.white_plane.copy()      # ← 加这行
        new_env.black_plane = self.black_plane.copy()      # ← 加这行
        new_env.last_move_plane = self.last_move_plane.copy()  # ← 加这行
        return new_env