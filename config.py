import torch

class Config:
    """Attn-JGNN 配置参数"""

    # ======================== 数据参数 ========================
    num_vars = None
    num_clauses = None

    # ======================== 模型参数 ========================
    dim = 64                       # 特征维度
    max_iter = 5                   # 消息传递最大迭代次数
    num_heads_init = 4             # 初始注意力头数
    num_heads_max = 8              # 最大注意力头数
    head_increase_step = 1000      # 每 N 步增加 1 个头
    dropout = 0.1
    num_gat_layers = 2             # 两层 GAT
    num_mlp_layers = 1             # 一层 MLP

    # ======================== 损失参数 ========================
    lambda_constraint = 0.1        # 约束感知正则项权重

    # ======================== 训练参数 ========================
    lr = 1e-3
    epochs = 200
    batch_size = 1                 # 每公式一个图结构，逐个训练
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ======================== 树分解参数 ========================
    tree_decomposer_path = '/usr/bin/htd'
    tree_decomposer_name = 'htd'
    # FlowCutter 修改版可执行文件 (需在WSL中编译: make clean && make)
    flow_cutter_path = r'C:\Users\123456\Desktop\flow-cutter-pace17-master\flow-cutter-pace17-master\flow_cutter_pace17'
    # 如果未编译, 设置为此路径也会自动回退到 NetworkX (os.path.exists 检查失败)
    target_treewidth = 5

    # 自适应树宽档位
    tw_candidates = [3, 5, 8, 12]

    # 树分解复杂度权重
    w_rho = 0.4                    # 密度权重
    w_cycle = 0.3                  # 围长权重
    w_length = 0.3                 # 平均子句长度权重

    # 簇复杂度得分权重 (公式 3.21, 用于动态头分配)
    w_cluster_size = 0.5           # 簇规模权重
    w_cluster_density = 0.3        # 簇内约束密度权重
    w_cluster_connect = 0.2        # 簇间关联度权重

    # 自适应树分解所需的数据集统计量
    rho_max = 5.0                  # BIRD/SATLIB 参考最大密度
    L_max = 10.0                   # 参考最大平均子句长度
    cycle_norm_factor = 20.0       # 围长归一化因子

    # ======================== 任务类型 ========================
    task = 'sat'                   # 'sat' 或 'pb'