import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import seaborn as sns

# 设置后端，防止在某些服务器上报错
try:
    mpl.use('Agg') 
except:
    pass

class PolicyVisualizer:
    def __init__(self, save_dir='policy_vis', cmap='viridis'):
        """
        初始化可视化器
        :param save_dir: 图片保存的目录
        :param cmap: 颜色映射 ('viridis', 'plasma', 'inferno', 'magma' 等)
        """
        self.save_dir = save_dir
        self.cmap = cmap
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # 设置中文字体，防止乱码 (可选，如果有中文字体)
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

    def save_heatmap(self, data, table_name, epoch=None):
        """
        保存热图 (最适合查看概率分布)
        """
        data = self._to_numpy(data)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # 使用 seaborn 绘制热图，显示每个格子的颜色深浅（代表概率）
        sns.heatmap(data, annot=False, fmt=".3f", cmap=self.cmap, 
                    cbar=True, linewidths=0.5, linecolor='black', ax=ax)
        
        title = f"Policy Heatmap: {table_name}"
        if epoch is not None:
            title += f" | Epoch {epoch}"
        ax.set_title(title)
        
        # 反转 Y 轴，让 (0,0) 在左上角，符合围棋/五子棋矩阵习惯
        ax.invert_yaxis()
        
        save_path = os.path.join(self.save_dir, f"{table_name}_heatmap.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_3d_bar_grid(self, data, table_name, epoch=None):
        """
        保存伪 3D 柱状图 (兼容所有 matplotlib 版本)
        每个格子就是一个柱子，颜色代表高度(概率)
        """
        data = self._to_numpy(data)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # 创建 x, y 坐标网格
        rows, cols = data.shape
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # --- 生成颜色数组 ---
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=data.min(), vmax=data.max())
        cmap = plt.cm.get_cmap(self.cmap)
        
        # 计算每个数据的颜色值
        # cmap 返回 shape (N, 4)，是 RGBA 数组
        colors = cmap(norm(data.flatten()))
        
        # --- 关键修复：转换为 Python 列表 ---
        # Matplotlib 的 bar 对 color 参数传入 numpy 数组时有时会报错
        # 转换为 .tolist() 是最稳妥的解决办法
        colors_list = colors.tolist()
        
        # 绘制柱状图
        ax.bar(x.flatten(), data.flatten(), width=0.8, 
               color=colors_list,          # ✅ 使用列表
               edgecolor='white', 
               linewidth=0.2)
        
        # 手动添加 colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Probability')
        
        # 设置坐标轴刻度
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.set_xticklabels(np.arange(cols))
        ax.set_yticklabels(np.arange(rows))
        
        # 反转 Y 轴
        ax.invert_yaxis()
        
        title = f"Policy 3D-Bar: {table_name}"
        if epoch is not None:
            title += f" | Epoch {epoch}"
        ax.set_title(title)
        
        save_path = os.path.join(self.save_dir, f"{table_name}_3dbar.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


    def save_side_by_side(self, pred_data, target_data, table_name, epoch=None):
        """
        将 网络预测 和 MCTS目标 并排画出来对比
        """
        pred_data = self._to_numpy(pred_data)
        target_data = self._to_numpy(target_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：预测
        sns.heatmap(pred_data, annot=False, cmap='Blues', cbar=True, linewidths=0.5, ax=ax1, vmin=0, vmax=1.0)
        ax1.set_title(f"Network Prediction")
        ax1.invert_yaxis()
        
        # 右图：目标
        sns.heatmap(target_data, annot=False, cmap='Reds', cbar=True, linewidths=0.5, ax=ax2, vmin=0, vmax=1.0)
        ax2.set_title(f"MCTS Target")
        ax2.invert_yaxis()
        
        fig.suptitle(f"{table_name} | Epoch {epoch if epoch else ''}", fontsize=14)
        
        save_path = os.path.join(self.save_dir, f"{table_name}_compare.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _to_numpy(self, data):
        if torch.is_tensor(data):
            return data.cpu().detach().numpy()
        return np.array(data)
    
    def save_board_with_heatmap(self, board_data, probs_data, table_name, last_move=None):
        """
        绘制棋盘并叠加概率热力图
        
        Args:
            board_data: Tensor or Numpy Array, shape [2, 15, 15]
                        通道 0 表示当前方棋子，通道 1 表示对手棋子
            probs_data: Tensor or Numpy Array, shape [225] or [15, 15]
                        网络预测的概率分布
            table_name: str, 保存文件名
            last_move: tuple or None, (row, col) 上一步落子位置，将高亮显示
        """
        board = self._to_numpy(board_data)
        probs = self._to_numpy(probs_data).reshape(15, 15)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # 1. 绘制概率热力图作为背景
        # 使用 seaborn，center=0 会把中间值设为浅色，不合适；
        # 我们希望概率大的地方亮，小的地方暗。使用 'viridis' 或 'YlOrRd'
        sns.heatmap(probs, annot=False, cmap=self.cmap, cbar=True, 
                    linewidths=0.5, linecolor='black', ax=ax,
                    vmin=0, vmax=probs.max()) # 动态调整最大值，让热力对比更明显
        
        # 2. 绘制棋子
        # board[0] 是当前方（比如黑），board[1] 是对手（比如白）
        rows, cols = np.where(board[0] == 1) # 黑棋位置
        ax.scatter(cols, rows, c='black', s=150, marker='o', edgecolors='gray', label='Current Player', zorder=2)
        
        rows_opp, cols_opp = np.where(board[1] == 1) # 白棋位置
        ax.scatter(cols_opp, rows_opp, c='white', s=150, marker='o', edgecolors='black', label='Opponent', zorder=2)
        
        # 3. 高亮上一步 (亮色圈圈)
        if last_move:
            r, c = last_move
            # 画一个更大的空心圆圈
            ax.scatter(c, r, facecolors='none', edgecolors='#00FF00', s=400, linewidths=3, zorder=3)
            # 画一个实心小点
            ax.scatter(c, r, facecolors='#00FF00', s=20, zorder=3)

        # 4. 设置坐标轴
        ax.set_xticks(np.arange(15))
        ax.set_yticks(np.arange(15))
        ax.set_xticklabels(np.arange(15))
        ax.set_yticklabels(np.arange(15))
        ax.invert_yaxis() # 反转 Y 轴，左上角为 (0,0)
        
        ax.set_title(f"{table_name} | Prob Overlay")
        
        save_path = os.path.join(self.save_dir, f"{table_name}_board.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_raw_board(self, board_data, table_name, last_move=None):
        """
        仅绘制当前棋盘状态（不带热力图），用于查看棋局本身
        """
        board = self._to_numpy(board_data)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # 绘制背景网格（米色或浅灰色，模仿棋盘）
        ax.set_facecolor('#F0D9B5') 
        
        rows, cols = np.where(board[0] == 1)
        ax.scatter(cols, rows, c='black', s=200, marker='o', edgecolors='gray', zorder=2)
        
        rows_opp, cols_opp = np.where(board[1] == 1)
        ax.scatter(cols_opp, rows_opp, c='white', s=200, marker='o', edgecolors='black', zorder=2)
        
        if last_move:
            r, c = last_move
            ax.scatter(c, r, facecolors='none', edgecolors='red', s=400, linewidths=3, zorder=3)
            ax.scatter(c, r, facecolors='red', s=30, zorder=3)

        ax.set_xticks(np.arange(15))
        ax.set_yticks(np.arange(15))
        ax.set_xticklabels(np.arange(15))
        ax.set_yticklabels(np.arange(15))
        ax.grid(True, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.invert_yaxis()
        
        # 隐藏刻度数字，只保留网格
        ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
        
        ax.set_title(f"Raw Board: {table_name}")
        
        save_path = os.path.join(self.save_dir, f"{table_name}_raw.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
