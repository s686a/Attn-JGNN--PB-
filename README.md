[README.md](https://github.com/user-attachments/files/29173583/README.md)
# Attn-JGNN: Attention-Enhanced Join-Graph Neural Networks

> **论文**: 结合注意力机制和连接图神经网络的#SAT求解方法研究  
> **英文**: Attn-JGNN — Attention + Join-GNN for #SAT Solving  
> **作者**: 张籍新 | 吉林大学 | 2025  
> **GPU**: NVIDIA A100 (80GB) | **Python**: 3.9+

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 目录结构](#2-目录结构)
- [3. 环境安装](#3-环境安装)
- [4. 数据准备](#4-数据准备)
- [5. 模型训练](#5-模型训练)
- [6. 模型评估](#6-模型评估)
- [7. 基线实验](#7-基线实验)
- [8. 所有关键参数](#8-所有关键参数)
- [9. 所有脚本说明](#9-所有脚本说明)
- [10. 文件功能速查表](#10-文件功能速查表)
- [11. 常见问题与注意事项](#11-常见问题与注意事项)

---

## 1. 项目概述

Attn-JGNN 是面向 **#SAT (命题模型计数)** 和 **PB模型计数 (Pseudo-Boolean Model Counting)** 的图神经网络求解框架。核心创新：

| 模块 |功能 |
|------|------|
| **IJGP 消息传递** | 树分解→连接图，替代BP算法，避免环结构消息重复 |
| **分层注意力** |簇内注意力(捕捉局部约束) + 簇间注意力(全局一致性) |
| **约束感知模块** |损失函数正则化，显式惩罚违反子句的赋值 |
| **动态注意力头** |训练步长增长 + 按簇复杂度差异化分配 |
| **Bethe-Join自由能** |适配连接图结构的Bethe自由能 → logZ估计 |
| **PB扩展 (Attn-JGNN-PB)** |加权连接图 + 系数感知注意力 + 软约束损失 |

**评估指标**: 对数模型数的 RMSE (Root Mean Square Error)

**核心结果** (`build.sh`):
| 实验 | 数据集 | RMSE |
|------|--------|------|
| SAT / BIRD | BIRD (8类) | 1.15 |
| SAT / SATLIB | SATLIB (5类) | 1.18 |
| PB / Combined | PB06+ExactCover+WSN | 0.19 |
| 消融-基线 | JGNN 无注意力 | 1.33 |
| 消融-分层注意力 | +Hierarchical Attention | 1.26 |
| 消融-约束感知 | +Constraint Loss | 1.19 |
| 消融-动态注意力 | +Dynamic Heads (Full) | 1.15 |

---

## 2. 目录结构

```
Attn-JGNN-/
├── models/                     #   核心模型代码
│   ├── attn_jgnn.py            #   主模型 AttnJGNN (SAT + PB)
│   ├── ijgp_layers.py          #   簇内/簇间IJGP消息传递 
│   ├── hierarchical_attn.py    #   分层注意力封装
│   ├── bethe_free_energy.py    #   Bethe-Join自由能估计 
│   ├── updater.py              #   GRU节点特征更新
│   └── gat_layer.py            #   参考实现 (已整合到ijgp_layers)
│
├── graph/                      #   图构建
│   ├── factor_graph.py         #   因子图 / 加权因子图构建
│   ├── join_graph.py           #   连接图构建 (树分解→簇划分)
│   └── tree_decomposition.py   #   自适应树分解 (复杂度感知选树宽)
│
├── losses/                     #   损失函数
│   ├── constraint_loss.py      #   SAT硬约束损失 
│   └── pb_constraint_loss.py   #   PB软约束损失 
│
├── data/                       #   数据处理
│   ├── cnf_parser.py           #   DIMACS CNF解析
│   ├── pb_parser.py            #   OPB/WPBF解析 + 标准化 
│   ├── pb_to_cnf.py            #   PB→CNF转换 (Warners编码 + BDD编码)
│   ├── PB_Competition2006/     #   PB竞赛2006 (287实例, train/test)
│   ├── ExactCover/             #   精确覆盖 (539实例, train/test)
│   ├── WSN/                    #   无线传感器网络 (600实例, train/test)
│   ├── BIRD/                   #   BIRD SAT基准
│   ├── SATLIB/                 #   SATLIB基准
│   └── labels/                 #   真值标签 (.pkl)
│
├── utils/                      #   工具
│   ├── dynamic_heads.py        #   动态注意力头分配器 
│   ├── flow_cutter_utils.py    #   FlowCutter外部工具调用
│   ├── metrics.py              #   RMSE等评估指标
│   ├── dataset.py              #   旧版BP数据集格式 (NSNet/BPNN用)
│   ├── dataloader.py           #   旧版BP数据加载器
│   ├── solvers.py              #   外部求解器封装 (DSHARP/ApproxMC3/F2等)
│   ├── options.py              #   旧版参数 (已废弃)
│   ├── utils.py                #   CNF预处理工具 (旧版)
│   └── logger.py               #   日志双写工具
│
├── baselines/                  #   基线实验 
│   ├── nsnet_model/            #   NSNet模型复制 (避免包冲突)
│   ├── nsnet_runner.py         #   NSNet PB基线运行器
│   ├── approxmc_pb_runner.py   #   ApproxMC-PB封装
│   └── run_all_baselines.py    #   统一基线编排 (6组对比)
│
├── experiments/                #   消融实验与效率对比
│   ├── count_attn_computations.py  # 注意力计算次数对比
│   ├── final_exp.py                # 综合实验
│   └── *.txt                       # 实验输出日志
│
├── logs/                       #   训练日志
├── checkpoints/                #   模型检查点
├── results/                    #   评估结果
├── plots/                      #   图表输出
│
├── train_sat.py                # SAT批量训练脚本
├── train_pb.py                 # PB批量训练脚本
├── train_single.py             # 单实例训练脚本 (调试用)
├── evaluate.py                 # 统一评估脚本
├── config.py                   # 全局配置参数
├── build.sh                    # 一键实验流水线
├── requirement.txt             # Python依赖
│
├── generate_labels.py          # 真值标签生成 (调用DSHARP/PBMC/PBCount)
├── generate_3-sat_data.py      # 3-SAT随机数据生成
├── generate_ca_data.py         # CA数据生成
├── download_bird_data.py       # BIRD数据集下载
├── download_satlib_data.py     # SATLIB数据集下载
├── examples_sat.py             # 50个手工SAT测试实例
├── examples_pb.py              # 50个手工PB测试实例
└── framework_plot.py           # 论文框架图生成
```

---

## 3. 环境安装

### 3.1 Python 环境

```bash
# 创建虚拟环境 
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirement.txt
```

**核心依赖**:
| 包 | 版本 | 
|----|------|------|
| `torch` | >=2.0.0 |  
| `torch-geometric` | >=2.5.0 |  
| `torch-scatter` | >=2.1.0 |  
| `numpy` | >=1.24.0 |  
| `networkx` | >=3.0 |  
| `matplotlib` | >=3.7.0 |  
| `tqdm` | >=4.65.0 |  
| `python-sat` | >=1.8.0 |  

### 3.2 外部工具

| 工具 | 用途 | 安装方式 |
|------|------|---------|
| **FlowCutter** | 树分解 | 下载可执行文件, 配置 `config.flow_cutter_path` |
| **DSHARP** | SAT精确计数(标签生成) | Linux编译 → 通过WSL调用 |
| **PBMC** | PB精确计数(标签生成) | Linux编译 → 通过WSL调用 |
| **PBCount** | PB精确计数(交叉验证) | Linux编译 → 通过WSL调用 |
| **ApproxMC** | 传统近似计数对比 | `git clone https://github.com/meelgroup/approxmc` |

**FlowCutter 配置**:
```python
# config.py 第32行
flow_cutter_path = r'C:\Users\123456\Desktop\flow-cutter-pace17-master'
# 或 Linux: '/usr/local/bin/flow_cutter'
```

### 3.3 WSL (Windows Subsystem for Linux) 环境配置

> DSHARP、PBMC、PBCount 均为 Linux C++ 编译的二进制文件, 无法直接在 Windows 上运行。
> 通过 WSL2 在 Windows 内运行完整 Linux 内核, 编译求解器后, 从 Python 通过 `wsl <命令>` 调用。

#### 3.3.1 安装 WSL2 + Ubuntu

以**管理员身份**打开 PowerShell:

```powershell
# 1. 启用 WSL 功能
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 2. 重启计算机
restart-computer

# 3. 重启后, 安装 WSL2 内核更新包
#    下载: https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi
#    双击安装

# 4. 设置 WSL2 为默认版本
wsl --set-default-version 2

# 5. 安装 Ubuntu 24.04
wsl --install -d Ubuntu-24.04
```

首次启动 Ubuntu 时按提示设置 Linux 用户名和密码。

#### 3.3.2 安装 Linux 编译依赖

进入 WSL Ubuntu 终端:

```bash
# 更新包管理器
sudo apt update && sudo apt upgrade -y

# 安装编译工具链
sudo apt install -y build-essential cmake g++ make

# 安装数学库 (GMP, MPFR — PBMC/PBCount 依赖)
sudo apt install -y libgmp-dev libmpfr-dev libboost-all-dev

# 安装其他工具
sudo apt install -y git curl wget unzip

# 验证安装
g++ --version      # 应显示 13.x 或更高
cmake --version    # 应显示 3.x 或更高
```

#### 3.3.3 编译 DSHARP (SAT 精确计数器)

```bash
# 在 WSL Ubuntu 中执行
cd ~
mkdir -p solvers && cd solvers

# 方案A: 从源码编译 (推荐)
git clone https://github.com/QuMuLab/dsharp.git
cd dsharp
make
# 编译成功后生成 ./dsharp 可执行文件
# 记下路径: /home/<用户名>/solvers/dsharp/dsharp
```

> 如果源码编译失败, 可从 [DSharp releases](https://github.com/QuMuLab/dsharp/releases) 下载预编译二进制, 放置于 `~/solvers/dsharp/`。

#### 3.3.4 编译 PBMC (PB 精确计数器)

```bash
# 在 WSL Ubuntu 中执行
cd ~/solvers

# 克隆 PBMC 仓库
git clone https://github.com/meelgroup/pbmc.git
cd pbmc

# 编译 (一键脚本)
bash setup-complete-code.sh
bash build.sh rs

# 编译成功后生成 ./bin/pbmc 可执行文件
# 记下路径: /home/<用户名>/solvers/pbmc/bin/pbmc
```

#### 3.3.5 编译 PBCount 

```bash
# 在 WSL Ubuntu 中执行
cd ~/solvers

# 克隆 PBCount 仓库
git clone https://github.com/meelgroup/pbcount.git
cd pbcount

# 编译
bash COMPILE.sh

# 编译成功后生成 ./pbcount 可执行文件
# 记下路径: /home/<用户名>/solvers/pbcount/pbcount
```

#### 3.3.6 验证 WSL 编译结果

```bash
# 在 WSL Ubuntu 中执行
cd ~/solvers

# 验证 DSHARP
./dsharp/dsharp --help 2>&1 | head -3

# 验证 PBMC
./pbmc/bin/pbmc --help 2>&1 | head -3

# 验证 PBCount
./pbcount/pbcount --help 2>&1 | head -3
```

#### 3.3.7 配置 Windows → WSL 调用路径

从 Windows 的 Python 调用 WSL 中的求解器, 需要使用 `wsl` 命令前缀。创建 Windows 批处理包装脚本:

**`external/dsharp.bat`** (在项目根目录下创建 `external/` 文件夹):
```batch
@echo off
wsl ~/solvers/dsharp/dsharp %*
```

**`external/pbmc.bat`**:
```batch
@echo off
wsl ~/solvers/pbmc/bin/pbmc %*
```

**`external/pbcount.bat`**:
```batch
@echo off
wsl ~/solvers/pbcount/pbcount %*
```

`generate_labels.py` 已内置 `--wsl` 标志, 自动处理路径转换。使用方式:

```bash
# === 通过 WSL 生成标签 (SAT) ===
# DSHARP 的 Linux 路径为 ~/solvers/dsharp/dsharp
python generate_labels.py \
    --solver dsharp \
    --data_dir ./data/BIRD/train \
    --solver_path ~/solvers/dsharp/dsharp \
    --output ./data/labels/bird_labels.pkl \
    --wsl

# === 通过 WSL 生成标签 (PB, 使用 PBMC) ===
python generate_labels.py \
    --solver pbmc \
    --data_dir ./data/PB_Competition2006/test \
    --solver_path ~/solvers/pbmc/bin/pbmc \
    --output ./data/labels/pb_labels.pkl \
    --wsl

# === 交叉验证 (使用 PBCount 对10%子集验证) ===
python generate_labels.py \
    --solver pbcount \
    --data_dir ./data/PB_Competition2006/test \
    --solver_path ~/solvers/pbcount/pbcount \
    --output ./data/labels/pb_labels_crossval.pkl \
    --wsl

# === 生成全部三个PB数据集的标签 ===
for ds in PB_Competition2006 ExactCover WSN; do
    python generate_labels.py \
        --solver pbmc \
        --data_dir ./data/$ds/test \
        --solver_path ~/solvers/pbmc/bin/pbmc \
        --output ./data/labels/${ds}_labels.pkl \
        --wsl
done
```

> 自动将 Windows 路径 (`C:\Users\...`) 转换为 WSL 路径 (`/mnt/c/Users/...`), 并在命令前添加 `wsl` 前缀。无需手动创建 `.bat` 包装脚本。

#### 3.3.8 WSL 调试技巧

```bash
# 测试 WSL 是否正常工作
wsl echo "WSL OK"

# 测试 DSHARP 是否可调用
wsl ~/solvers/dsharp/dsharp --help

# 路径转换: Windows路径 -> WSL路径
# C:\Users\123456\Desktop\test.cnf  ->  /mnt/c/Users/123456/Desktop/test.cnf

# 在 Python 中调用 WSL 求解器示例
import subprocess
result = subprocess.run(
    ['wsl', '~/solvers/dsharp/dsharp',
     '/mnt/c/Users/123456/Desktop/test.cnf',
     '-F', '-cnf', '-noPP'],
    capture_output=True, text=True, timeout=5000
)
```

### 3.4 FlowCutter 树宽控制

> 可通过人工控制树分解的树宽度灵活调整连接图复杂度"。
> FlowCutter 原版是 PACE 2017 竞赛的启发式求解器, **不原生支持目标树宽参数** — 它在无限循环中用不同随机种子不断搜索更优分解, 直到超时或被 SIGINT 终止。

#### 3.4.1 修改 C++ 源码

- FlowCutter 输出的是合法的**树分解 (tree decomposition)**, 必须满足 **running intersection property** (连通性): 包含同一变量的所有簇构成一棵连通子树
- **Python 层面不能后处理** — 拆分/合并簇会破坏此性质, 导致 IJGP 消息传递错误
- 修改 C++ 源码, 在 FlowCutter 的搜索过程中, 一旦找到满足目标宽度的分解就**立即退出**

#### 3.4.2 修改内容 (3 处, 仅 `src/pace.cpp`)

| # | 位置 (行号) | 修改 | 说明 |
|---|-----------|------|------|
| 1 | 第35行 `best_bag_size` 之后 | `int target_bag_size = INT_MAX;` | 全局变量, 存储目标树宽对应的 bag size, `INT_MAX` 表示不限制 |
| 2 | 第397-411行 参数解析 | 将 `if(argc==3)` 改为 `for` 循环 | 同时解析 `--target-width N` 和 `-s seed`; 原版只支持 `-s` |
| 3 | 第456行 `best_bag_size=tw` 之后 | `if(best_bag_size <= target_bag_size) signal_handler(0);` | 找到满足条件的分解时立即输出并退出 |

**详细代码 diff**:

```diff
 // line 34-35
 int best_bag_size = numeric_limits<int>::max();
+int target_bag_size = numeric_limits<int>::max();

 // line 397-411 (替换原来的 if(argc==3) 分支)
-int random_seed = 0;
-if(argc == 3){
-    if(string(argv[1]) == "-s"){
-        random_seed = atoi(argv[2]);
-    }
-}
+int random_seed = 0;
+for(int i = 1; i < argc; ++i){
+    if(string(argv[i]) == "--target-width" && i+1 < argc){
+        int tw = atoi(argv[i+1]);
+        target_bag_size = tw + 1;
+    }
+    if(string(argv[i]) == "-s" && i+1 < argc){
+        random_seed = atoi(argv[i+1]);
+    }
+}

 // line 456-463 (在 best_bag_size = tw; delete[]old_decomposition; 之后)
 best_bag_size = tw;
 delete[]old_decomposition;
+
+if(best_bag_size <= target_bag_size){
+    signal_handler(0);
 }
```

#### 3.4.3 编译

项目目录 `flow-cutter-pace17-master/` 下已是修改后的源码。在 **WSL Ubuntu** 中编译:

```bash
# 进入源码目录
cd /mnt/c/Users/123456/Desktop/flow-cutter-pace17-master/flow-cutter-pace17-master

# 安装编译依赖 (仅首次需要)
sudo apt install -y g++ make

# 编译 (Makefile 内容: g++ -Wall -std=c++11 -O3 -DNDEBUG src/*.cpp -o flow_cutter_pace17)
make clean && make

# 验证生成的可执行文件
ls -lh flow_cutter_pace17
# -rwxr-xr-x 1 user user 约2MB flow_cutter_pace17
```

> **编译参数说明**: `-O3` 最高优化, `-DNDEBUG` 禁用断言 (大量 `assert` 会显著拖慢速度), `-std=c++11` C++11 标准。如果在其他 Linux 发行版编译报错, 将 `-std=c++11` 改为 `-std=c++17` 再试。

#### 3.4.4 使用方式

```bash
# === 无目标限制 (行为与原版完全相同) ===
./flow_cutter_pace17 input.gr
# 一直运行, 每30秒输出当前最优树宽, 直到 SIGINT/Ctrl+C

# === 限制树宽 ≤ 5 ===
./flow_cutter_pace17 --target-width 5 input.gr
# 找到树宽 ≤ 5 的分解立即退出, 可能在数秒内完成

# === 限制树宽带随机种子 ===
./flow_cutter_pace17 --target-width 8 -s 42 input.gr
```

**输出格式** (PACE `.td` 标准):

```
c status 5 1234              ← 每30秒输出: 当前最优bag_size 时间(ms)
c target treewidth set to 5 (bag size 6)   ← --target-width 被识别
s td 10 5 4                  ← 树分解头部: 10 bags, 树宽5, 节点数
b 1 2 3                      ← bag 1: 包含节点 1,2,3
b 2 3 4                      ← bag 2
...
```

#### 3.4.5 Python 侧自动调用

已集成到项目代码中, 无需手动操作:

```
select_treewidth(公式复杂度) → target_tw ∈ {3,5,8,12}
        ↓
run_flow_cutter(G, timeout=30, target_treewidth=target_tw)
        ↓
cmd = [flow_cutter_path, '--target-width', str(target_tw), input.gr]
```

如果 FlowCutter 是未修改的原版 (不支持 `--target-width`), 该参数会被**静默忽略** (原版只解析它认识的 `-s`), 行为与原来一致, 不会报错。

#### 3.4.6 注意事项

| # | 注意点 | 说明 |
|---|--------|------|
| 1 | **不能 Python 后处理** | 对 FlowCutter 输出的簇进行拆分/合并会破坏 tree decomposition 的 running intersection property, 导致消息传递错误 |
| 2 | **target-width 是最优条件, 非精确匹配** | `--target-width 5` 表示"找到树宽 ≤ 5 时退出", 可能找到树宽 3 的结果。并非要求树宽恰好为 5 |
| 3 | **小图可能立即退出** | 对于节点数 < target_width 的图, FlowCutter 可能瞬间退出 (初始单簇已满足条件) |
| 4 | **大图可能需要较长时间** | 对于复杂图 (如 BIRD 的 DQMR), 即使设 target_width=12, 也可能运行数十秒才找到满足条件的分解 |
| 5 | **signal_handler 保证输出** | 使用 `signal_handler(0)` 而非 `exit(0)`, 确保在退出前输出当前最优分解。这是 FlowCutter 原有的信号处理机制 |
| 6 | **编译需 WSL/Linux** | FlowCutter 是 Linux C++ 程序, 需在 WSL Ubuntu 中编译。Windows 上无法直接编译或运行 |
| 7 | **原版兼容** | 不传 `--target-width` 时, `target_bag_size = INT_MAX`, 条件永远不满足, 行为与原版完全一致 |
| 8 | **自适应树宽选择** | `tree_decomposition.py` 中的 `select_treewidth()` 根据公式密度/围长/子句长度自动从 {3,5,8,12} 中选择目标树宽 |

---

### 4.1 SAT 数据集 (BIRD + SATLIB)

```bash
# 下载 BIRD 数据集
python download_bird_data.py

# 下载 SATLIB 数据集
python download_satlib_data.py

# 生成真值标签 (调用 DSharp, 超时 5000s)
python generate_labels.py --solver dsharp \
    --data_dir ./data/BIRD/train \
    --solver_path ./external/dsharp \
    --output ./data/labels/bird_labels.pkl
```

**数据格式**: DIMACS CNF (`.cnf`)

```
p cnf 3 2
1 2 0
-1 3 0
```

### 4.2 PB 数据集 (PB06 + ExactCover + WSN)

数据集已预置在 `data/` 下, 按 70/30 随机划分:

```bash
# 查看数据统计
ls data/PB_Competition2006/train/ | wc -l  # 201
ls data/PB_Competition2006/test/  | wc -l  # 87
ls data/ExactCover/train/         | wc -l  # 377
ls data/ExactCover/test/          | wc -l  # 162
ls data/WSN/train/                | wc -l  # 420
ls data/WSN/test/                 | wc -l  # 180
```

**数据格式**: Weighted PB Format (`.wpbf`)

```
* #variable= 5 #constraint= 2
+2 x1 +1 x2 >= 2 ;
+1 x1 +1 x3 <= 1 ;
```

### 4.3 生成 PB 真值标签

```bash
# 使用 PBMC (推荐)
python generate_labels.py --solver pbmc \
    --data_dir ./data/PB_Competition2006/test \
    --solver_path ./external/pbmc \
    --output ./data/labels/pb_labels.pkl

# 使用 PBCount (交叉验证)
python generate_labels.py --solver pbcount \
    --data_dir ./data/PB_Competition2006/test \
    --solver_path ./external/pbcount \
    --output ./data/labels/pb_labels.pkl
```

---

## 5. 模型训练

### 5.1 SAT 模型训练

```bash
# BIRD 数据集 (70/30 划分)
python train_sat.py \
    --dataset bird \
    --data_dir ./data/BIRD/train \
    --label_file ./data/labels/bird_labels.pkl \
    --epochs 200 \
    --lr 1e-3 \
    --save_dir ./checkpoints

# SATLIB 数据集 (60/20/20 划分)
python train_sat.py \
    --dataset satlib \
    --data_dir ./data/SATLIB \
    --label_file ./data/labels/satlib_labels.pkl \
    --epochs 200 \
    --lr 1e-3 \
    --save_dir ./checkpoints
```

**训练过程输出示例**:
```
Attn-JGNN SAT | Dataset: bird
Device: cuda | Epochs: 200 | LR: 0.001
Train: 140, Test: 60
Parameters: 253,962
Epoch   0 | Train RMSE: 2.1432 | Test RMSE: 2.0891 | Time: 12s
Epoch  50 | Train RMSE: 1.3201 | Test RMSE: 1.2845 | Time: 245s
Epoch 100 | Train RMSE: 1.1892 | Test RMSE: 1.1630 | Time: 478s
...
Done! Best Test RMSE: 1.1542 | Total: 945s
```

### 5.2 PB 模型训练

```bash
# 全部三个 PB 数据集联合训练
python train_pb.py \
    --epochs 200 \
    --lr 1e-3 \
    --seed 42 \
    --save_dir ./checkpoints
```

训练脚本自动加载 `data/PB_Competition2006/`、`data/ExactCover/`、`data/WSN/` 下的 train/test 数据。

### 5.3 单实例训练 

```bash
python train_single.py \
    --task sat \
    --input ./data/BIRD/train/DQMR/instance.cnf \
    --true_logz 10.5 \
    --epochs 200
```

### 5.4 一键流水线

```bash
# 运行全部实验
bash build.sh all

# 仅 SAT 实验
bash build.sh sat

# 仅 PB 实验
bash build.sh pb

# 仅消融实验
bash build.sh ablation
```

---

## 6. 模型评估

### 6.1 SAT 评估

```bash
python evaluate.py \
    --task sat \
    --data_dir ./data/BIRD/test \
    --label_file ./data/labels/bird_test_labels.pkl \
    --checkpoint ./checkpoints/attn_jgnn_bird_best.pt \
    --output ./results/sat_bird_eval.txt
```

### 6.2 PB 评估

```bash
python evaluate.py \
    --task pb \
    --data_dir ./data/PB_Competition2006/test \
    --label_file ./data/labels/pb_labels.pkl \
    --checkpoint ./checkpoints/attn_jgnn_pb_best.pt \
    --output ./results/pb_eval.txt
```

**评估输出示例**:
```
Attn-JGNN Evaluation | Task: sat
Instances evaluated: 60
RMSE:               1.1542
Within +-0.5:       42.50%
Within +-1.0:       68.33%
Mean Relative Error: 0.3210
Total time:          3.2s
Avg time/instance:   0.053s
```

---

## 7. 基线实验 

### 7.1 PB→CNF + NSNet 基线

```bash
# 需要先训练 NSNet (或使用预训练权重)
python baselines/nsnet_runner.py \
    --pb_data ./data/PB_Competition2006/test \
    --encoding both \
    --nsnet_checkpoint ./checkpoints/nsnet_mc_best.pt \
    --output ./results/nsnet_baseline.json
```

**注意**: NSNet 代码位于 `C:\Users\123456\Desktop\NSNet-main\NSNet-main\src\`, 模型文件已复制到 `baselines/nsnet_model/` 避免包冲突。NSNet 使用 BPG 图格式, 与 Attn-JGNN 的因子图格式**不兼容**, 由 `nsnet_runner.py` 自动转换。

### 7.2 ApproxMC-PB 基线

```bash
python baselines/approxmc_pb_runner.py \
    --pb_data ./data/WSN/test \
    --approxmc_pb_path ./external/approxmc \
    --output ./results/approxmc_baseline.json \
    --timeout 5000
```

如果 ApproxMC 不支持原生 PB 格式, 自动回退为 **PB→CNF(Warners)→ApproxMC**。

### 7.3 一键运行全部基线

```bash
python baselines/run_all_baselines.py \
    --data_dir ./data \
    --attn_jgnn_checkpoint ./checkpoints/attn_jgnn_pb_best.pt \
    --nsnet_checkpoint ./checkpoints/nsnet_mc_best.pt \
    --approxmc_pb_path ./external/approxmc \
    --output_dir ./results
```

输出 6 组对比的 RMSE 表格:
```
================================================================================
RMSE (对数计数)
================================================================================
方法                                  Solved   RMSE      Time(s)
-----------------------------------------------------------------
Attn-JGNN-PB                          429      0.1900    2.3
PB->CNF(warners)+Attn-JGNN            429      0.2700    3.1
PB->CNF(bdd)+Attn-JGNN                429      0.2500    2.8
PB->CNF(warners)+NSNet                429      0.5600    4.5
PB->CNF(bdd)+NSNet                    429      0.5100    4.2
ApproxMC-PB                           420      0.0410    98.6
================================================================================
```

---

## 8. 所有关键参数

### 8.1 `config.py` 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **模型超参** |
| `dim` | 64 | 节点特征维度 (32/64/128) |
| `max_iter` | 5 | 消息传递迭代次数 |
| `num_heads_init` | 4 | 初始注意力头数 |
| `num_heads_max` | 8 | 最大注意力头数 |
| `head_increase_step` | 1000 | 每N训练步增加1个头 |
| `dropout` | 0.1 | Dropout比率 |
| `num_mlp_layers` | 1 | MLP隐藏层数 |
| **损失参数** | | |
| `lambda_constraint` | 0.1 | 约束正则项权重 |
| **训练参数** | | |
| `lr` | 1e-3 | 学习率 |
| `epochs` | 200 | 训练轮数 |
| `batch_size` | 1 | 批大小 (每公式图结构不同) |
| `device` | `cuda`/`cpu` | 自动检测 |
| **树分解参数** | | |
| `flow_cutter_path` | (需配置) | FlowCutter可执行文件路径 |
| `tw_candidates` | [3,5,8,12] | 自适应树宽候选值 |
| `w_rho/w_cycle/w_length` | 0.4/0.3/0.3 | 树宽选择权重 |
| `w_cluster_size/density/connect` | 0.5/0.3/0.2 | 簇复杂度权重|

### 8.2 命令行参数

**`train_sat.py`**:
| 参数 | 必需 | 说明 |
|------|------|------|
| `--dataset` | ✓ | `bird` 或 `satlib` |
| `--data_dir` | ✓ | CNF文件目录 |
| `--label_file` | | 标签pickle文件 |
| `--epochs` | | 训练轮数 (默认200) |
| `--lr` | | 学习率 (默认1e-3) |
| `--seed` | | 随机种子 (默认42) |
| `--save_dir` | | 检查点保存目录 |

**`train_pb.py`**:
| 参数 | 必需 | 说明 |
|------|------|------|
| `--epochs` | | 训练轮数 (默认200) |
| `--lr` | | 学习率 (默认1e-3) |
| `--seed` | | 随机种子 (默认42) |
| `--save_dir` | | 检查点保存目录 |

**`evaluate.py`**:
| 参数 | 必需 | 说明 |
|------|------|------|
| `--task` | ✓ | `sat` 或 `pb` |
| `--data_dir` | ✓ | 测试数据目录 |
| `--checkpoint` | ✓ | 模型检查点 `.pt` |
| `--label_file` | | 标签pickle文件 |
| `--output` | | 结果输出文件 |

---

## 9. 所有脚本说明

### 核心脚本

| 脚本 | 用途 | 用法 |
|------|------|------|
| `train_sat.py` | SAT模型训练 (BIRD/SATLIB) | `python train_sat.py --dataset bird --data_dir ./data/BIRD/train` |
| `train_pb.py` | PB模型训练 (3数据集联合) | `python train_pb.py --epochs 200` |
| `train_single.py` | 单实例训练 (调试用) | `python train_single.py --task sat --input file.cnf --true_logz 10.5` |
| `evaluate.py` | 统一评估 (SAT/PB) | `python evaluate.py --task pb --data_dir ./data/WSN/test --checkpoint ckpt.pt` |

### 数据处理

| 脚本 | 用途 | 用法 |
|------|------|------|
| `data/cnf_parser.py` | DIMACS CNF解析 | `from data.cnf_parser import parse_cnf` |
| `data/pb_parser.py` | OPB/WPBF解析 + 标准化 | `from data.pb_parser import parse_opb, normalize_pb_constraints` |
| `data/pb_to_cnf.py` | PB→CNF (Warners + BDD) | `from data.pb_to_cnf import warners_encode, bdd_encode` |
| `generate_labels.py` | 真值标签生成 | `python generate_labels.py --solver dsharp --data_dir ...` |

### 基线实验

| 脚本 | 用途 | 用法 |
|------|------|------|
| `baselines/nsnet_runner.py` | PB→CNF+NSNet基线 | `python baselines/nsnet_runner.py --pb_data ... --nsnet_checkpoint ...` |
| `baselines/approxmc_pb_runner.py` | ApproxMC-PB基线 | `python baselines/approxmc_pb_runner.py --pb_data ... --approxmc_pb_path ...` |
| `baselines/run_all_baselines.py` | 一键6组基线对比 | `python baselines/run_all_baselines.py --data_dir ...` |

### 工具脚本

| 脚本 | 用途 |
|------|------|
| `build.sh` | 一键实验流水线 (环境→编译→数据→标签→训练→消融) |
| `generate_3-sat_data.py` | 随机3-SAT数据生成 |
| `generate_ca_data.py` | CA数据生成 |
| `download_bird_data.py` | BIRD数据集下载 |
| `download_satlib_data.py` | SATLIB数据集下载 |
| `examples_sat.py` | 50个手工SAT测试实例 |
| `examples_pb.py` | 50个手工PB测试实例 |
| `framework_plot.py` | 论文框架图 (图3.1) 生成 |
| `experiments/count_attn_computations.py` | 注意力计算次数对比实验 (§3.6.4) |

---

## 10. 文件功能速查表

### models/ (核心模型)

| 文件 | 关键类/函数 |
|------|------------|
| `attn_jgnn.py` | `AttnJGNN` — forward, init_node_feats, compute_factor_values |
| `ijgp_layers.py` | `IntraClusterIJGP` , `InterClusterIJGP` |
| `hierarchical_attn.py` | `HierarchicalAttention` — 组装簇内+簇间 |
| `bethe_free_energy.py` | `BetheFreeEnergy` , (1-d_v)加权 |
| `updater.py` | `IJGPUpdater` — GRU(msg, hidden) |
| `gat_layer.py` | `CoefficientAwareGATLayer` (已整合到ijgp_layers) |

### graph/ (图构建)

| 文件 | 关键函数 |
|------|---------|
| `factor_graph.py` |  `build_factor_graph` — SAT因子图 / PB加权因子图 |
| `join_graph.py` | `build_join_graph`, `build_join_graph_adaptive` |
| `tree_decomposition.py` | `adaptive_tree_decomposition` — 复杂度感知选树宽, `select_treewidth` |

### losses/ (损失函数)

| 文件 | 关键类 |
|------|---------|--------|
| `constraint_loss.py` |  `SATConstraintLoss` — sigmoid软满足 + -log惩罚 |
| `pb_constraint_loss.py` |  `PBConstraintLoss` — 系数感知 + 软约束 |

### utils/ (工具)

| 文件 | 功能 | 
|------|------|
| `dynamic_heads.py` | 动态头分配  | 
| `flow_cutter_utils.py` | FlowCutter调用封装 | 
| `metrics.py` | RMSE, 相对误差, 达标率 | 
| `dataset.py` | BPG/LCG图格式 (NSNet/BPNN用) | 
| `dataloader.py` | 旧版DataLoader | 
| `solvers.py` | 外部求解器封装 (DSHARP/ApproxMC3/F2) |
| `options.py` | 旧版参数 | 
| `utils.py` | CNF解析 + 预处理 (旧版) | 
| `logger.py` | 日志双写 (文件+终端) | 

### baselines/ (基线)

| 文件 |  功能 |
|------|------|
| `nsnet_model/nsnet.py` |  NSNet模型 (从NSNet项目复制) |
| `nsnet_model/mlp.py` | NSNet依赖的MLP层 |
| `nsnet_runner.py` | PB→CNF+NSNet推理封装 |
| `approxmc_pb_runner.py` | ApproxMC-PB封装 (PB直解 + CNF回退) |
| `run_all_baselines.py` | 6组基线统一编排 |

### data/ (数据)

| 文件/目录 | 说明 |
|-----------|------|
| `cnf_parser.py` | DIMACS CNF解析 |
| `pb_parser.py` | OPB/WPBF解析 + 标准化 (统一>=, 系数正化, 归一化) |
| `pb_to_cnf.py` | Warners编码 (顺序计数器) + BDD编码 (ROBDD→Tseitin) |
| `PB_Competition2006/` | PB竞赛2006: train 201 + test 87 |
| `ExactCover/` | 精确覆盖: train 377 + test 162 |
| `WSN/` | 无线传感器网络: train 420 + test 180 |
| `BIRD/` | BIRD SAT基准 (8类) |
| `SATLIB/` | SATLIB基准 (5类) |
| `labels/` | 真值标签 .pkl 文件 |

---

## 11. 常见问题与注意事项

### Q1: FlowCutter 相关问题

**Q1a: 报错 `FlowCutter executable not found`**
**A**: 修改 `config.py` 第32行 `flow_cutter_path`。若 FlowCutter 不可用, 代码自动回退到 NetworkX 启发式, 功能正常但树宽不受控。

**Q1b: 树宽如何控制?**
**A**: FlowCutter 原版不支持。本项目已修改 `src/pace.cpp` (3处改动), 新增 `--target-width N` 参数。需在 WSL 中重新编译: `make clean && make`。详见 [§3.4](#34-flowcutter-树宽控制-修改版)。

**Q1c: 为什么不能 Python 后处理 FlowCutter 的输出?**
**A**: 拆分簇会破坏 tree decomposition 的 running intersection property (包含同一变量的所有簇必须构成连通子树), 破坏后 IJGP 消息传递会出错。因此必须在 C++ 搜索层面控制树宽。

### Q2: PB 训练加载不到数据

**A**: 检查 `train_pb.py` 第92行是否同时支持 `.opb` 和 `.wpbf`:
```python
if not (fn.endswith('.opb') or fn.endswith('.wpbf')):
```
确保数据文件扩展名为 `.wpbf`。

### Q3: 模型参数数量

| 任务 | 参数量 |
|------|--------|
| SAT | 253,962 |
| PB | 258,698 (多一个 `bound_proj` 层) |
| NSNet | 58,434 |

### Q4: 训练集/测试集划分

| 数据集 | 划分方式 | 比例 |
|--------|---------|------|
| BIRD | 随机划分 | 70/30 |
| SATLIB | 随机划分 | 60/20/20 (含验证集) |
| PB (全部3个) | 随机划分 (seed=42) | 70/30 |

### Q5: `sys.path` 与 NSNet 导入冲突

NSNet 的 `models/` 包与 Attn-JGNN 的 `models/` 包同名。解决方式: NSNet 模型已复制到 `baselines/nsnet_model/`, 通过 `from baselines.nsnet_model.nsnet import NSNet` 导入。

### Q6: 旧版代码文件

以下文件来自早期 BP-based 实现 (NSNet/BPNN), 与当前 Attn-JGNN 的 IJGP+因子图架构**不兼容**, 仅作参考:

- `utils/dataset.py` — BPG/LCG图格式
- `utils/dataloader.py` — 旧版DataLoader
- `utils/solvers.py` — 旧版求解器封装
- `utils/utils.py` — 旧版CNF预处理
- `utils/options.py` — 旧版参数
- `train_pb_quick.py` — 废弃的快速训练脚本


