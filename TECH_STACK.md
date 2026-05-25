# 五子棋强化学习项目技术栈总结

## 1. 项目定位

本项目是一个基于 Python 的五子棋（Wuziqi/Gomoku）强化学习项目。整体实现接近 AlphaZero 思路：使用神经网络同时预测策略和局面价值，再结合蒙特卡洛树搜索（MCTS）进行自博弈、训练和推理。

项目包含以下能力：

- 五子棋规则环境与棋盘状态管理
- PyTorch 策略-价值神经网络
- MCTS 搜索与 AI 落子选择
- 自博弈训练与经验回放
- 预训练数据生成与加载
- Pygame 本地图形界面
- 模型 checkpoint、训练日志和 replay memory 管理

## 2. 运行环境

| 项目 | 技术 | 依据 |
| --- | --- | --- |
| 编程语言 | Python | 项目主体由 `.py` 文件组成 |
| Python 版本 | Python 3.11 | `.python-version` 为 `3.11`，`pyproject.toml` 要求 `>=3.11` |
| 项目元数据 | `pyproject.toml` | 声明项目名 `wuziqi`、版本 `0.1.0` |
| 配置格式 | YAML | `hyperparameters.yaml` 管理训练与推理参数 |
| 运行平台 | 本地脚本项目 | 无 Web 服务或前后端分离结构 |

当前 `pyproject.toml` 中 `dependencies = []`，实际依赖没有被写入项目元数据，需要根据源码手动安装。

## 3. 主要第三方依赖

| 依赖 | 作用 | 主要使用文件 |
| --- | --- | --- |
| PyTorch (`torch`) | 定义神经网络、训练、推理、模型保存和加载 | `network.py`, `main.py`, `mcts.py` |
| NumPy (`numpy`) | 棋盘矩阵、状态通道、批量数据、数据增强和样本生成 | `wuziqi_env.py`, `main.py`, `mcts.py`, `dataset_kill.py` |
| PyYAML (`yaml`) | 读取训练超参数配置 | `main.py` |
| Pygame (`pygame`) | 绘制棋盘、棋子、按钮和 AI 对局窗口 | `wuziqi_gui.py` |

建议补充的安装依赖：

```bash
pip install torch numpy pyyaml pygame
```

如果使用 GPU 训练，`torch` 应按本机 CUDA 版本安装对应发行包。

## 4. Python 标准库使用

| 标准库 | 用途 |
| --- | --- |
| `argparse` | 命令行参数解析 |
| `os` | 路径拼接、目录创建 |
| `pickle` | 保存和加载数据集、经验回放 |
| `datetime`, `time` | 训练计时、日志时间 |
| `random` | 随机采样、数据生成 |
| `copy` | 状态复制 |
| `threading` | GUI 中异步执行 AI 思考 |
| `collections.deque` | 经验回放队列 |
| `typing` | 类型标注 |

## 5. 核心模块划分

| 文件 | 模块职责 | 技术点 |
| --- | --- | --- |
| `wuziqi_env.py` | 五子棋环境 | 棋盘矩阵、合法落子、胜负判断、撤销、状态通道 |
| `network.py` | 神经网络模型 | PyTorch、卷积层、BatchNorm、残差块、policy head、value head |
| `mcts.py` | 蒙特卡洛树搜索 | TreeNode、批量推理、PUCT/探索项、策略回传 |
| `main.py` | 训练与推理入口 | Agent、自博弈、优化器、预训练、checkpoint 管理 |
| `experience_replay.py` | 经验回放 | `deque` 存储、随机 batch 采样 |
| `wuziqi_gui.py` | 图形界面 | Pygame 绘制、点击交互、AI 对局线程 |
| `dataset_kill.py` | 训练样本生成 | NumPy 向量化、唯一制胜点样本、pickle 输出 |
| `hyperparameters.yaml` | 实验配置 | 多组实验参数、模型路径、训练开关 |

## 6. 模型与算法栈

### 策略-价值网络

`network.py` 中的 `Network` 继承自 `torch.nn.Module`，结构包括：

- 输入卷积层：处理棋盘多通道状态。
- 残差塔：由多个 `Residual_block` 组成，默认配置中常用 `10` 个残差块。
- 策略头：输出每个棋盘位置的落子 logits。
- 价值头：输出当前局面的胜负价值，最后使用 `tanh` 限制到 `[-1, 1]`。

默认状态输入为 `8` 个通道，包含黑棋、白棋、历史落子平面和当前玩家信息。

### MCTS

`mcts.py` 实现蒙特卡洛树搜索：

- 使用 `TreeNode` 保存访问次数、分数、先验概率和子节点。
- 通过神经网络批量评估叶子节点。
- 对非法或无效落子进行 mask。
- 训练时在根节点策略中加入 Dirichlet 噪声，增强探索。
- 根据访问次数生成训练用 policy target。

### 训练方式

`main.py` 中的 `Agent` 负责训练流程：

- 从 `hyperparameters.yaml` 读取实验配置。
- 初始化环境、网络、MCTS 和 replay memory。
- 通过自博弈采集 `(state, policy, value)`。
- 使用 PyTorch 优化 policy loss 和 value loss。
- 使用 `AdamW` 作为主要优化器。
- 支持预训练、继续训练、保存最佳模型和保存经验池。

## 7. 数据与持久化

| 类型 | 文件示例 | 说明 |
| --- | --- | --- |
| 模型 checkpoint | `*.pt` | PyTorch `state_dict` 保存的模型权重 |
| 经验回放 | `*_memory.pkl` | pickle 保存的 replay memory |
| 预训练数据 | `standard1_dataset.pkl`, `basic_skill*.pkl` | 用于预训练或战术训练的数据 |
| 训练日志 | `*.txt`, `*.log` | 训练过程、优化过程和调试日志 |
| 实验目录 | `runs/` | 不同版本模型和日志的输出目录 |

## 8. 配置体系

项目使用 `hyperparameters.yaml` 管理多组实验，例如：

- `alphaTao-v0.9`
- `alphaTao-v0.4-run`
- `test1`
- `test_overfit`
- `test2`

主要配置项包括：

| 配置项 | 含义 |
| --- | --- |
| `board_size` | 棋盘大小 |
| `residual_blocks` | 残差块数量 |
| `search_num` | 每步 MCTS 搜索次数 |
| `self_play_num` | 自博弈局数 |
| `replay_memory_size` | 经验回放容量 |
| `optimizer_batch_size` | 训练 batch size |
| `inference_batch_size` | MCTS 推理 batch size |
| `lr` | 学习率 |
| `run_dir`, `restore_dir` | 模型、日志和 memory 保存路径 |
| `is_pretrain` | 是否启用预训练 |
| `is_training` | 是否训练 |
| `is_self_play` | 是否自博弈 |
| `is_continue_training` | 是否从已有模型继续训练 |

## 9. GUI 技术栈

`wuziqi_gui.py` 使用 Pygame 实现桌面 GUI：

- `WuziqiGUI`: 基础五子棋界面，支持玩家点击落子、重置和胜负提示。
- `AiMatch`: 人机/AI 对局界面，调用 MCTS 生成 AI 落子。
- 使用 `threading.Thread` 将 AI 思考放到后台线程，避免界面完全阻塞。

## 10. 常用入口

训练指定实验：

```bash
python main.py alphaTao-v0.9 --train
```

加载模型并进入 AI 对局/推理模式：

```bash
python main.py alphaTao-v0.9
```

启动基础 Pygame GUI：

```bash
python wuziqi_gui.py
```

生成战术样本数据：

```bash
python dataset_kill.py
```

## 11. 当前工程特点

- 项目是脚本式 Python 工程，尚未拆分为标准 Python package。
- 强化学习主流程集中在 `main.py`，功能较完整但文件较大。
- 算法路线是 MCTS + policy/value network + self-play。
- 支持 CUDA 自动检测：`torch.device("cuda" if torch.cuda.is_available() else "cpu")`。
- 模型、数据集和日志直接保存在仓库目录或 `runs/` 下。
- README 和部分中文注释存在编码损坏，需要统一修复为 UTF-8。

## 12. 工程化建议

1. 在 `pyproject.toml` 中补齐实际依赖：

```toml
dependencies = [
    "numpy",
    "pyyaml",
    "pygame",
    "torch",
]
```

2. 将训练产物加入 `.gitignore`：

```gitignore
runs/
*.pt
*.pkl
*.log
```

3. 拆分 `main.py`：

- `agent.py`: Agent 和训练流程
- `trainer.py`: 优化与 loss 计算
- `pretrain.py`: 预训练逻辑
- `cli.py`: 命令行入口

4. 增加测试覆盖：

- `WuziqiEnv` 的胜负判断、非法落子、撤销逻辑
- MCTS 的合法动作 mask 和 policy 生成
- `dataset_kill.py` 生成样本的合法性

5. 修复文档和源码注释编码，统一使用 UTF-8。
