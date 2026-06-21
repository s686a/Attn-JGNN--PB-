import torch
import torch.nn as nn
from torch_scatter import scatter_sum


class BetheFreeEnergy(nn.Module):
    """
    Bethe-Join 自由能估计 

    F_Bethe = Σ_α H(C_α) + Σ_v (1 - d_v) H(v)         

    其中:
      H(C_α) = MLP_cluster( pool({h_i | i ∈ C_α}) )     
      H(v)   = MLP_var( h_v )                             
      d_v    = 变量 v 所在的簇数                           

    log Z ≈ -F_Bethe                                      
    """

    def __init__(self, dim, num_mlp_layers=2):
        super().__init__()

        # 簇联合熵 MLP 
        cluster_layers = []
        for _ in range(num_mlp_layers):
            cluster_layers.append(nn.Linear(dim, dim))
            cluster_layers.append(nn.ReLU())
        cluster_layers.append(nn.Linear(dim, 1))
        self.cluster_mlp = nn.Sequential(*cluster_layers)

        # 变量局部熵 MLP — 独立于簇MLP
        var_layers = []
        for _ in range(num_mlp_layers):
            var_layers.append(nn.Linear(dim, dim))
            var_layers.append(nn.ReLU())
        var_layers.append(nn.Linear(dim, 1))
        self.var_mlp = nn.Sequential(*var_layers)

    def forward(self, node_feats, num_vars, clusters, cluster_vars, cluster_cons):
        """
        参数:
            node_feats:    [N, dim]            所有节点特征
            num_vars:      int                 变量数
            clusters:      list[list]          每簇节点列表
            cluster_vars:  list[list]          每簇变量列表
            cluster_cons:  list[list]          每簇约束列表

        返回:
            logZ: 标量，配分函数的对数 ≈ -F_Bethe
        """
        var_feats = node_feats[:num_vars]
        device = node_feats.device
        K = len(clusters)

        # ===== 簇联合熵 H(C_α) =====
        cluster_energy = []
        for cid in range(K):
            vars_in = cluster_vars[cid]
            cons_in = cluster_cons[cid]
            # 簇内所有节点 (变量 + 约束)
            indices = vars_in + [num_vars + c for c in cons_in]
            if indices:
                cluster_x = node_feats[indices].mean(dim=0, keepdim=True)
                H_c = self.cluster_mlp(cluster_x).squeeze()
            else:
                H_c = torch.tensor(0.0, device=device)
            cluster_energy.append(H_c)
        cluster_energy = torch.stack(cluster_energy)  # [K]

        # ===== 变量局部熵 H(v) × (1-d_v)=====
        var_energy_raw = self.var_mlp(var_feats).squeeze(-1)  # [V]

        # 计算每个变量的度数 d_v (所在簇数)
        var_degrees = torch.zeros(num_vars, device=device)
        for cid in range(K):
            if cluster_vars[cid]:
                v_idx = torch.tensor(cluster_vars[cid], device=device)
                var_degrees[v_idx] += 1

        # (1 - d_v) 加权: 避免重复计数
        # 对于 d_v = 1 的变量, 权重为 0 (其熵完全包含在簇联合熵中)
        # 对于 d_v > 1 的变量, 权重为负 (修正过度计数)
        degree_weight = 1.0 - var_degrees
        var_energy = (var_energy_raw * degree_weight).sum()

        # ===== Bethe 自由能 → logZ =====
        total_energy = cluster_energy.sum() + var_energy
        logZ = -total_energy
        return logZ