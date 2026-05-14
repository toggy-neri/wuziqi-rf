# 五子棋游戏 - Wuziqi Game

一个简单的五子棋游戏，支持pygame图形界面和强化学习训练。

## 文件结构

- `wuziqi_env.py` - 游戏环境类，包含游戏逻辑和RL训练接口
- `wuziqi_gui.py` - pygame可视化界面
- `main.py` - 主程序入口

## 安装依赖

```bash
conda activate dqn
pip install pygame numpy
```

## 运行方式

### 1. 图形界面模式

```bash
python main.py --mode gui
```

或者直接运行：
```bash
python wuziqi_gui.py
```

### 2. 测试环境模式

```bash
python main.py --mode test
```

## RL训练接口

`WuziqiEnv` 类提供了标准的强化学习接口：

```python
from wuziqi_env import WuziqiEnv

env = WuziqiEnv(board_size=15)

# 重置环境
state = env.reset()

# 执行动作
action = (row, col)  # 落子位置
next_state, reward, done, info = env.step(action)

# 获取合法动作
valid_moves = env.get_valid_moves()

# 获取当前状态
state = env.get_state()

# 复制环境
env_copy = env.copy()
```

### 接口说明

- `reset()`: 重置游戏，返回初始状态
- `step(action)`: 执行动作，返回 (next_state, reward, done, info)
  - action: (row, col) 元组，表示落子位置
  - reward: 1.0 表示黑棋胜利，-1.0 表示白棋胜利，0 表示继续，-10 表示非法移动
  - done: 游戏是否结束
  - info: 包含额外信息的字典
- `get_valid_moves()`: 返回所有合法落子位置的列表
- `get_state()`: 返回当前棋盘状态的副本
- `copy()`: 创建环境的深拷贝

### 状态表示

- 棋盘状态是一个 15x15 的 numpy 数组
- 0: 空位
- 1: 黑棋
- -1: 白棋

## 游戏规则

- 黑棋先手，白棋后手
- 先连成5子的一方获胜
- 棋盘填满则为平局

## 后续RL训练建议

1. 可以使用 DQN、PPO、AlphaZero 等算法进行训练
2. 状态空间：15x15 的棋盘
3. 动作空间：225个可能的落子位置（15x15）
4. 奖励设计：
   - 胜利：+1（黑棋）或 -1（白棋）
   - 非法移动：-10
   - 平局：0
   - 继续游戏：0

5. 可以考虑添加：
   - 位置编码
   - 历史状态（最近几步棋）
   - 对称性增强（旋转、镜像）
