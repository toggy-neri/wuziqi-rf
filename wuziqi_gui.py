import pygame
import sys
import threading

from typing import Tuple, Optional
from wuziqi_env import WuziqiEnv
from mcts import MCTS, TreeNode

class WuziqiGUI:

    def __init__(self, board_size: int = 15, cell_size: int = 40):
        pygame.init()
        
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = 50
        
        self.window_size = (
            self.board_size * self.cell_size + 2 * self.margin,
            self.board_size * self.cell_size + 2 * self.margin + 60
        )
        
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("五子棋 - Wuziqi")
        
        self.colors = {
            'background': (220, 179, 92),
            'line': (0, 0, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'highlight': (255, 0, 0),
            'text': (0, 0, 0),
            'button': (100, 150, 200),
            'button_hover': (120, 170, 220)
        }
        
        self.env = WuziqiEnv(board_size)
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 28)
        
        self.button_rect = pygame.Rect(
            self.window_size[0] // 2 - 60,
            self.window_size[1] - 50,
            120, 35
        )
        
    def draw_board(self):
        self.screen.fill(self.colors['background'])
        
        for i in range(self.board_size):
            start_x = self.margin
            end_x = self.margin + (self.board_size - 1) * self.cell_size
            y = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, self.colors['line'], 
                           (start_x, y), (end_x, y), 1)
        
        for i in range(self.board_size):
            start_y = self.margin
            end_y = self.margin + (self.board_size - 1) * self.cell_size
            x = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, self.colors['line'], 
                           (x, start_y), (x, end_y), 1)
        
        center = self.board_size // 2
        center_pos = self.margin + center * self.cell_size
        pygame.draw.circle(self.screen, self.colors['line'], 
                          (center_pos, center_pos), 4)
        
        star_points = [(3, 3), (3, 11), (11, 3), (11, 11)]
        for row, col in star_points:
            if row < self.board_size and col < self.board_size:
                x = self.margin + col * self.cell_size
                y = self.margin + row * self.cell_size
                pygame.draw.circle(self.screen, self.colors['line'], (x, y), 3)
    
    def draw_pieces(self):
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.env.board[row, col]
                if piece != 0:
                    x = self.margin + col * self.cell_size
                    y = self.margin + row * self.cell_size
                    color = self.colors['black'] if piece == 1 else self.colors['white']
                    pygame.draw.circle(self.screen, color, (x, y), self.cell_size // 2 - 2)
                    
                    if piece == -1:
                        pygame.draw.circle(self.screen, self.colors['line'], 
                                         (x, y), self.cell_size // 2 - 2, 1)
        
        if self.env.last_move:
            action = self.env.last_move 
            row = action // self.env.board_size
            col = action % self.env.board_size
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            pygame.draw.circle(self.screen, self.colors['highlight'], (x, y), 5)
    
    def draw_ui(self):
        y_offset = self.margin + self.board_size * self.cell_size + 10
        
        if self.env.done:
            if self.env.winner == 1:
                text = " Black Wins!"
            elif self.env.winner == -1:
                text = "White Wins!"
            else:
                text = " Draw!"
        else:
            player = "Black" if self.env.current_player == 1 else "White"
            text = f"{player}"
        
        text_surface = self.font.render(text, True, self.colors['text'])
        text_rect = text_surface.get_rect(center=(self.window_size[0] // 2, y_offset + 15))
        self.screen.blit(text_surface, text_rect)
        
        mouse_pos = pygame.mouse.get_pos()
        button_color = self.colors['button_hover'] if self.button_rect.collidepoint(mouse_pos) else self.colors['button']
        pygame.draw.rect(self.screen, button_color, self.button_rect, border_radius=5)
        
        button_text = self.small_font.render("reset", True, self.colors['white'])
        button_text_rect = button_text.get_rect(center=self.button_rect.center)
        self.screen.blit(button_text, button_text_rect)
    
    def screen_to_board(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        x, y = pos
        col = round((x - self.margin) / self.cell_size)
        row = round((y - self.margin) / self.cell_size)
        
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return (row, col)
        return None
    
    def handle_click(self, pos: Tuple[int, int]) -> bool:
        if self.button_rect.collidepoint(pos):
            self.env.reset()
            return False
        
        if self.env.done:
            return False
        
        board_pos = self.screen_to_board(pos)
        if board_pos:
            row, col = board_pos
            action = row * self.env.board_size + col
            if self.env._is_valid_move(row, col):
                self.env.step(action)
                return True
        
        return False
    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(event.pos)
            
            self.draw_board()
            self.draw_pieces()
            self.draw_ui()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()

class AiMatch:
    def __init__(self,env,network,search_num,batch_size):
        pygame.init()
        self.board_size = env.board_size
        self.env = env
        self.margin = 50
        self.cell_size = 40

        self.search_num = search_num
        self.batch_size = batch_size

        self.window_size = (
            self.board_size * self.cell_size + self.margin * 2,
            self.board_size * self.cell_size + self.margin * 4
        )


        self.screen = pygame.display.set_mode(self.window_size)

        pygame.display.set_caption("Wuziqi AI Match")

        self.colors = {
            'background': (220, 179, 92),
            'line': (0, 0, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'highlight': (255, 0, 0),
            'text': (0, 0, 0),
            'button': (100, 150, 200),
            'button_hover': (120, 170, 220),
            'thinking': (200, 80, 80),   
        }


        self.mcts = MCTS(network,self.env,False)
        self.mcts_ai = MCTS(network,self.env, False)
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None,24)

        self.button_rect = pygame.Rect(self.window_size[0] // 2 - 60, self.window_size[1]-50, 120, 35)

        self.is_ai_thinking = False
        self.ai_thread = None   
        self.pending_move = None

        self.human_player = 1  #todo 设置为-1会导致下不了棋
        self.root = TreeNode(parent=None)

        self.env.reset()
    
    def screen_to_board(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        x, y = pos
        col = round((x - self.margin) / self.cell_size)
        row = round((y - self.margin) / self.cell_size)
        
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return (row, col)
        return None
    
    def draw_board(self):
        self.screen.fill(self.colors['background'])
        for i in range(self.board_size):
            y = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, self.colors['line'],
                             (self.margin, y),
                             (self.margin + (self.board_size - 1) * self.cell_size, y), 1)
        for i in range(self.board_size):
            x = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, self.colors['line'],
                             (x, self.margin),
                             (x, self.margin + (self.board_size - 1) * self.cell_size), 1)
        center = self.board_size // 2
        cp = self.margin + center * self.cell_size
        pygame.draw.circle(self.screen, self.colors['line'], (cp, cp), 4)
        for row, col in [(3,3),(3,11),(11,3),(11,11)]:
            if row < self.board_size and col < self.board_size:
                pygame.draw.circle(self.screen, self.colors['line'],
                                   (self.margin + col * self.cell_size,
                                    self.margin + row * self.cell_size), 3)

    def draw_pieces(self):
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.env.board[row,col]
                if piece != 0:
                    x = self.margin + col * self.cell_size
                    y = self.margin + row * self.cell_size
                    color = self.colors['black'] if piece == 1 else self.colors['white']
                    pygame.draw.circle(self.screen, color, (x,y), self.cell_size // 2 - 2)
                    if piece == -1:
                        pygame.draw.circle(self.screen, self.colors['line'], (x,y), self.cell_size // 2 ,1 )

        if self.env.last_move is not None:
            row = self.env.last_move // self.board_size
            col = self.env.last_move % self.board_size
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            pygame.draw.circle(self.screen, self.colors['highlight'], (x,y), self.cell_size // 2 - 2)
            
    def draw_ui(self):
        y_offset = self.margin + self.board_size * self.cell_size + 10

        if self.env.done:
            if self.env.winner == self.human_player:
                text = "Human Win"
            elif self.env.winner == -self.human_player:
                text = "AI Win"
            else:
                text = "Draw"
            color = self.colors['text']
        elif self.is_ai_thinking:
            text = "AI Thinking"
            color = self.colors['thinking']
        else:
            text = "Human Turn"
            color = self.colors['text']

        text_surface = self.font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(self.window_size[0] // 2, y_offset+15))
        self.screen.blit(text_surface, text_rect)

        mouse_pos = pygame.mouse.get_pos()
        btn_color = self.colors['button_hover'] if self.button_rect.collidepoint(mouse_pos) else self.colors['button']
        pygame.draw.rect(self.screen, btn_color, self.button_rect, border_radius=5)
        btn_text = self.small_font.render("Reset", True, self.colors['white'])
        self.screen.blit(btn_text, btn_text.get_rect(center=self.button_rect.center))

    def reset(self):
        if self.ai_thread is not None and self.ai_thread.is_alive():
            return
        self.env.reset()
        self.root = TreeNode(parent=None)
        self.is_ai_thinking = False
        self.pending_move = None

    def trigger_ai(self):
        self.mcts_ai.env = self.env.copy()
        def _think():#inner function
            self.mcts_ai.search(self.root,self.batch_size,self.search_num,1.5)
            action , new_root = self.mcts_ai.choose(self.root,is_training=False)
            self.mcts_ai.env.step(action)
            self.root = new_root
            self.root.parent = None
            self.pending_move = action


        self.is_ai_thinking = True
        self.ai_thread = threading.Thread(target=_think, daemon=True)  #daemon thread, will be killed when main thread exits
        self.ai_thread.start()

    def handle_human_click(self, pos):
        if self.button_rect.collidepoint(pos):
            self.reset()
            return
        if self.env.done or self.is_ai_thinking:
            return
        if self.env.current_player != self.human_player:
            return
        board_pos = self.screen_to_board(pos)
        if board_pos is None:
            return

        row = board_pos[0]
        col = board_pos[1]
        if not self.env._is_valid_move(row,col):
            return
        if board_pos in self.root.children:
            self.root = self.root.children[board_pos]
            self.root.parent = None
        else:
            self.root = TreeNode(parent=None)

        _,_,done,_ = self.env.step(row*self.env.board_size+col)
        if not done:
            self.trigger_ai()

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_human_click(event.pos)

            if self.pending_move is not None and not self.env.done:
                self.env.step(self.pending_move)
                self.pending_move = None

                self.is_ai_thinking = False
            self.draw_board()
            self.draw_pieces()

            

            self.draw_ui()
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        sys.exit()
        


if __name__ == "__main__":
    game = WuziqiGUI(board_size=15, cell_size=40)
    game.run()
