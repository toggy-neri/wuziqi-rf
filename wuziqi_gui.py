import pygame
import sys
import threading
import json
import os
from datetime import datetime

from typing import Tuple, Optional
import numpy as np
import torch
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
    def __init__(self,env,network,search_num,batch_size,use_virtual_loss=False,virtual_loss=1.0):
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


        self.mcts = MCTS(
            network,
            self.env,
            False,
            use_virtual_loss=use_virtual_loss,
            virtual_loss=virtual_loss,
        )
        self.mcts_ai = MCTS(
            network,
            self.env,
            False,
            use_virtual_loss=use_virtual_loss,
            virtual_loss=virtual_loss,
        )
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None,24)

        button_y = self.window_size[1]-50
        button_width = 78
        button_gap = 10
        button_start = self.window_size[0] // 2 - (button_width * 6 + button_gap * 5) // 2
        self.prev_button_rect = pygame.Rect(button_start, button_y, button_width, 35)
        self.button_rect = pygame.Rect(button_start + button_width + button_gap, button_y, button_width, 35)
        self.next_button_rect = pygame.Rect(button_start + 2 * (button_width + button_gap), button_y, button_width, 35)
        self.mask_button_rect = pygame.Rect(button_start + 3 * (button_width + button_gap), button_y, button_width, 35)
        self.tree_button_rect = pygame.Rect(button_start + 4 * (button_width + button_gap), button_y, button_width, 35)
        self.fade_button_rect = pygame.Rect(button_start + 5 * (button_width + button_gap), button_y, button_width, 35)

        self.is_ai_thinking = False
        self.ai_thread = None   
        self.pending_move = None
        self.pending_ai_record = None

        self.human_player = 1
        self.ai_player = -self.human_player
        self.ai_player = -self.human_player
        self.root = TreeNode(parent=None)

        self.current_game_records = []
        self.last_game_records = []
        self.current_game_saved = False
        self.replay_active = False
        self.replay_index = 0
        self.use_policy_mask = True
        self.show_mcts_tree = False
        self.fade_replay_pieces = False
        self.replay_file = os.path.join("runs", "last_ai_match_replay.json")

        self.env.reset()
        self.load_last_game()
        self.start_ai_turn_if_needed()
    
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

    def get_display_state(self):
        if self.replay_active and self.last_game_records:
            if self.replay_index <= 0:
                board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
                return board, None, None
            record = self.last_game_records[self.replay_index - 1]
            board = np.array(record["board_after"], dtype=np.int8)
            return board, record["action"], record

        return self.env.board, self.env.last_move, None

    def draw_heatmap(self, record, board_after=None):
        if record is None:
            return
        policy = record.get("policy")
        if policy is None:
            return

        probs = np.array(policy, dtype=np.float32).reshape(self.board_size, self.board_size)
        board_before = np.array(record["board_before"], dtype=np.int8)
        if self.use_policy_mask:
            valid_mask = MCTS.get_valid_mask(board_before, radius=2).reshape(self.board_size, self.board_size)
            probs = probs * valid_mask
            prob_sum = float(probs.sum())
            if prob_sum <= 0:
                return
            probs = probs / prob_sum

        max_prob = float(probs.max())
        if max_prob <= 0:
            return

        if board_after is None:
            board_after = np.array(record["board_after"], dtype=np.int8)
        overlay = pygame.Surface(self.window_size, pygame.SRCALPHA)
        empty_radius = max(5, self.cell_size // 3)
        ring_radius = max(8, self.cell_size // 2 + 3)
        ring_width = 4

        for row in range(self.board_size):
            for col in range(self.board_size):
                if board_before[row, col] != 0:
                    continue
                prob = float(probs[row, col])
                if prob <= max_prob * 0.01:
                    continue
                strength = min(1.0, prob / max_prob)
                alpha = int(35 + 180 * strength)
                x = self.margin + col * self.cell_size
                y = self.margin + row * self.cell_size
                color = (255, 80, 25, alpha)
                if board_after[row, col] != 0:
                    pygame.draw.circle(overlay, color, (x, y), ring_radius, ring_width)
                else:
                    pygame.draw.circle(overlay, color, (x, y), empty_radius)

        self.screen.blit(overlay, (0, 0))

    def draw_mcts_heatmap(self, record, board_after=None):
        if not self.show_mcts_tree or record is None:
            return
        policy = record.get("mcts_policy")
        if policy is None:
            return

        probs = np.array(policy, dtype=np.float32).reshape(self.board_size, self.board_size)
        max_prob = float(probs.max())
        if max_prob <= 0:
            return

        board_before = np.array(record["board_before"], dtype=np.int8)
        if board_after is None:
            board_after = np.array(record["board_after"], dtype=np.int8)

        overlay = pygame.Surface(self.window_size, pygame.SRCALPHA)
        empty_radius = max(5, self.cell_size // 4)
        ring_radius = max(10, self.cell_size // 2 + 8)
        ring_width = 4

        for row in range(self.board_size):
            for col in range(self.board_size):
                if board_before[row, col] != 0:
                    continue
                prob = float(probs[row, col])
                if prob <= max_prob * 0.01:
                    continue
                strength = min(1.0, prob / max_prob)
                alpha = int(45 + 190 * strength)
                x = self.margin + col * self.cell_size
                y = self.margin + row * self.cell_size
                color = (40, 120, 255, alpha)
                if board_after[row, col] != 0:
                    pygame.draw.circle(overlay, color, (x, y), ring_radius, ring_width)
                else:
                    pygame.draw.circle(overlay, color, (x, y), empty_radius)

        self.screen.blit(overlay, (0, 0))

    def draw_pieces(self, board=None, last_move=None, faded=False):
        if board is None:
            board = self.env.board
        if last_move is None:
            last_move = self.env.last_move

        piece_layer = None
        if faded:
            piece_layer = pygame.Surface(self.window_size, pygame.SRCALPHA)

        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = board[row,col]
                if piece != 0:
                    x = self.margin + col * self.cell_size
                    y = self.margin + row * self.cell_size
                    color = self.colors['black'] if piece == 1 else self.colors['white']
                    target = piece_layer if faded else self.screen
                    draw_color = (*color, 135) if faded else color
                    pygame.draw.circle(target, draw_color, (x,y), self.cell_size // 2 - 2)
                    if piece == -1:
                        outline_color = (*self.colors['line'], 145) if faded else self.colors['line']
                        pygame.draw.circle(target, outline_color, (x,y), self.cell_size // 2 ,1 )

        if faded:
            self.screen.blit(piece_layer, (0, 0))

        if last_move is not None:
            row = last_move // self.board_size
            col = last_move % self.board_size
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            pygame.draw.circle(self.screen, self.colors['highlight'], (x,y), self.cell_size // 2 + 2, 4)
            
    def draw_ui(self):
        y_offset = self.margin + self.board_size * self.cell_size + 10

        if self.replay_active and self.last_game_records:
            record = None if self.replay_index == 0 else self.last_game_records[self.replay_index - 1]
            if record is None:
                text = f"Review 0/{len(self.last_game_records)}"
            else:
                row, col = divmod(record["action"], self.board_size)
                text = f"Review {self.replay_index}/{len(self.last_game_records)} {record['actor']} ({row},{col})"
            color = self.colors['text']
        elif self.env.done:
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
        self.draw_button(self.prev_button_rect, "Prev", self.can_replay(), mouse_pos)
        self.draw_button(self.button_rect, "Reset", True, mouse_pos)
        self.draw_button(self.next_button_rect, "Next", self.can_replay(), mouse_pos)
        mask_label = "Mask On" if self.use_policy_mask else "Raw"
        self.draw_button(self.mask_button_rect, mask_label, True, mouse_pos)
        tree_label = "Tree On" if self.show_mcts_tree else "Tree Off"
        self.draw_button(self.tree_button_rect, tree_label, True, mouse_pos)
        fade_label = "Fade On" if self.fade_replay_pieces else "Fade Off"
        self.draw_button(self.fade_button_rect, fade_label, True, mouse_pos)

    def draw_button(self, rect, label, enabled, mouse_pos):
        if not enabled:
            color = (135, 135, 135)
        else:
            color = self.colors['button_hover'] if rect.collidepoint(mouse_pos) else self.colors['button']
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        text = self.small_font.render(label, True, self.colors['white'])
        self.screen.blit(text, text.get_rect(center=rect.center))

    def can_replay(self):
        return bool(self.last_game_records)

    def reset(self):
        if self.ai_thread is not None and self.ai_thread.is_alive():
            return
        self.env.reset()
        self.root = TreeNode(parent=None)
        self.is_ai_thinking = False
        self.pending_move = None
        self.pending_ai_record = None
        self.current_game_records = []
        self.current_game_saved = False
        self.replay_active = False
        self.replay_index = 0
        self.start_ai_turn_if_needed()

    def start_ai_turn_if_needed(self):
        if not self.env.done and self.env.current_player != self.human_player:
            self.trigger_ai()

    def _advance_or_rebuild_root(self, action):
        if self.root is not None and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = TreeNode(parent=None)

    def get_network_policy(self, env):
        state = env.get_channel_state(env.board, env.current_player)
        inputs = torch.from_numpy(state).unsqueeze(0).float()
        try:
            model_device = next(self.mcts_ai.network.parameters()).device
            inputs = inputs.to(model_device)
        except StopIteration:
            pass

        was_training = self.mcts_ai.network.training
        self.mcts_ai.network.eval()
        with torch.no_grad():
            logits, _ = self.mcts_ai.network(inputs)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        if was_training:
            self.mcts_ai.network.train()

        valid_mask = (env.board.reshape(-1) == 0).astype(np.float32)
        probs = probs * valid_mask
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        return probs.astype(float).tolist()

    def build_move_record(self, actor, player, action, board_before, board_after, policy=None, mcts_policy=None):
        row, col = divmod(int(action), self.board_size)
        return {
            "step": len(self.current_game_records) + 1,
            "actor": actor,
            "player": int(player),
            "action": int(action),
            "position": [int(row), int(col)],
            "board_before": board_before.astype(int).tolist(),
            "board_after": board_after.astype(int).tolist(),
            "policy": policy,
            "mcts_policy": mcts_policy,
        }

    def record_move(self, record):
        self.current_game_records.append(record)
        if self.env.done:
            self.finalize_current_game()

    def finalize_current_game(self):
        if self.current_game_saved or not self.current_game_records:
            return

        self.last_game_records = [record.copy() for record in self.current_game_records]
        self.replay_active = True
        self.replay_index = len(self.last_game_records)
        self.current_game_saved = True
        self.save_last_game()

    def save_last_game(self):
        os.makedirs(os.path.dirname(self.replay_file), exist_ok=True)
        payload = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "board_size": self.board_size,
            "human_player": self.human_player,
            "winner": self.env.winner,
            "records": self.last_game_records,
        }
        with open(self.replay_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load_last_game(self):
        if not os.path.exists(self.replay_file):
            return
        try:
            with open(self.replay_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        if payload.get("board_size") != self.board_size:
            return
        records = payload.get("records", [])
        if isinstance(records, list):
            self.last_game_records = records
            self.replay_index = len(records)

    def step_replay(self, delta):
        if not self.last_game_records:
            return
        if self.ai_thread is not None and self.ai_thread.is_alive():
            return
        if not self.replay_active:
            self.replay_active = True
            self.replay_index = len(self.last_game_records) if delta < 0 else 0
        self.replay_index = max(0, min(len(self.last_game_records), self.replay_index + delta))

    def trigger_ai(self):
        self.mcts_ai.env = self.env.copy()
        def _think():#inner function
            board_before = self.mcts_ai.env.board.copy()
            player = self.mcts_ai.env.current_player
            policy = self.get_network_policy(self.mcts_ai.env)
            self.mcts_ai.search(self.root,self.batch_size,self.search_num,1.5)
            mcts_policy = self.mcts_ai.get_policy(self.root, self.board_size).astype(float).tolist()
            action , new_root = self.mcts_ai.choose(self.root,is_training=False)
            board_after, _, _, _ = self.mcts_ai.env.step(action)
            self.root = new_root
            self.root.parent = None
            self.pending_ai_record = self.build_move_record(
                "AI",
                player,
                action,
                board_before,
                board_after.copy(),
                policy,
                mcts_policy,
            )
            self.pending_move = action


        self.is_ai_thinking = True
        self.ai_thread = threading.Thread(target=_think, daemon=True)  #daemon thread, will be killed when main thread exits
        self.ai_thread.start()

    def handle_human_click(self, pos):
        if self.prev_button_rect.collidepoint(pos):
            self.step_replay(-1)
            return
        if self.next_button_rect.collidepoint(pos):
            self.step_replay(1)
            return
        if self.mask_button_rect.collidepoint(pos):
            self.use_policy_mask = not self.use_policy_mask
            return
        if self.tree_button_rect.collidepoint(pos):
            self.show_mcts_tree = not self.show_mcts_tree
            return
        if self.fade_button_rect.collidepoint(pos):
            self.fade_replay_pieces = not self.fade_replay_pieces
            return
        if self.button_rect.collidepoint(pos):
            self.reset()
            return
        if self.replay_active:
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

        action = row*self.env.board_size+col
        self._advance_or_rebuild_root(action)
        board_before = self.env.board.copy()
        player = self.env.current_player
        policy = self.get_network_policy(self.env)
        board_after,_,done,_ = self.env.step(action)
        self.record_move(self.build_move_record(
            "Human",
            player,
            action,
            board_before,
            board_after.copy(),
            policy,
        ))
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
                if self.pending_ai_record is not None:
                    self.record_move(self.pending_ai_record)
                self.pending_move = None
                self.pending_ai_record = None

                self.is_ai_thinking = False
            self.draw_board()
            board, last_move, replay_record = self.get_display_state()
            self.draw_heatmap(replay_record, board)
            self.draw_mcts_heatmap(replay_record, board)
            self.draw_pieces(board, last_move, faded=self.replay_active and self.fade_replay_pieces)

            

            self.draw_ui()
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        sys.exit()
        


if __name__ == "__main__":
    game = WuziqiGUI(board_size=15, cell_size=40)
    game.run()
