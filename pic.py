import math
import os

import matplotlib as mpl
import numpy as np

try:
    mpl.use('Agg')
except Exception:
    pass

import matplotlib.pyplot as plt


class PolicyVisualizer:
    def __init__(self, save_dir='policy_vis', cmap='viridis'):
        self.save_dir = save_dir
        self.cmap = cmap
        os.makedirs(self.save_dir, exist_ok=True)

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

    def save_policy_summary(self, pred_data, target_data, board_data, probs_data, table_name, last_move=None, epoch=None):
        pred = self._as_square(pred_data, 'pred_data')
        board_size = pred.shape[0]
        target = self._as_square(target_data, 'target_data', board_size)
        probs = self._as_square(probs_data, 'probs_data', board_size)
        board = self._to_numpy(board_data)

        if board.ndim != 3 or board.shape[0] < 2:
            raise ValueError('board_data must have shape [2, board_size, board_size] or more channels')
        if board.shape[-2:] != (board_size, board_size):
            raise ValueError('board_data spatial shape must match policy board size')

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        ax_pred, ax_target, ax_board = axes

        compare_vmax = max(float(pred.max()), float(target.max()), 1e-8)
        self._draw_heatmap(ax_pred, pred, 'Network Prediction', 'Blues', compare_vmax)
        self._draw_heatmap(ax_target, target, 'MCTS Target', 'Reds', compare_vmax)
        self._draw_heatmap(ax_board, probs, 'Board + Prediction Heatmap', self.cmap, max(float(probs.max()), 1e-8), alpha=0.78)
        self._draw_stones(ax_board, board, board_size)

        if last_move is not None:
            row, col = last_move
            ax_board.scatter(col, row, facecolors='none', edgecolors='#00ff66', s=420, linewidths=3, zorder=4)
            ax_board.scatter(col, row, c='#00ff66', s=22, zorder=4)

        title = table_name
        if epoch is not None:
            title = f'{title} | Epoch {epoch}'
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = os.path.join(self.save_dir, f'{table_name}_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path

    def _draw_heatmap(self, ax, data, title, cmap, vmax, alpha=1.0):
        image = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, origin='upper', alpha=alpha)
        ax.set_title(title)
        self._set_board_axes(ax, data.shape[0])
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    def _draw_stones(self, ax, board, board_size):
        stone_size = max(60, 240 - board_size * 6)

        rows, cols = np.where(board[0] == 1)
        ax.scatter(
            cols,
            rows,
            c='black',
            s=stone_size,
            marker='o',
            edgecolors='gray',
            linewidths=1.2,
            zorder=3,
        )

        rows, cols = np.where(board[1] == 1)
        ax.scatter(
            cols,
            rows,
            c='white',
            s=stone_size,
            marker='o',
            edgecolors='black',
            linewidths=1.2,
            zorder=3,
        )

    def _set_board_axes(self, ax, board_size):
        ax.set_xticks(np.arange(board_size))
        ax.set_yticks(np.arange(board_size))
        ax.set_xticklabels(np.arange(board_size))
        ax.set_yticklabels(np.arange(board_size))
        ax.set_xticks(np.arange(-0.5, board_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, board_size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.set_xlim(-0.5, board_size - 0.5)
        ax.set_ylim(board_size - 0.5, -0.5)

    def _as_square(self, data, name, board_size=None):
        data = self._to_numpy(data)

        if board_size is None:
            board_size = self._infer_board_size(data, name)

        if data.shape == (board_size, board_size):
            return data

        if data.size != board_size * board_size:
            raise ValueError(f'{name} cannot be reshaped to [{board_size}, {board_size}]')

        return data.reshape(board_size, board_size)

    def _infer_board_size(self, data, name):
        if data.ndim >= 2 and data.shape[-1] == data.shape[-2]:
            return data.shape[-1]

        board_size = int(math.sqrt(data.size))
        if board_size * board_size != data.size:
            raise ValueError(f'{name} must be square or flat square policy data')

        return board_size

    def _to_numpy(self, data):
        if hasattr(data, 'detach'):
            data = data.detach()
        if hasattr(data, 'cpu'):
            data = data.cpu()
        return np.asarray(data)
