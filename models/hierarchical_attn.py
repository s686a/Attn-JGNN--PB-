import torch
import torch.nn as nn
from .ijgp_layers import IntraClusterIJGP, InterClusterIJGP


class HierarchicalAttention(nn.Module):
    """
    IJGP 层次注意力模块
    支持运行时动态调整注意力头数。
    """

    def __init__(self, config, edge_feat_dim):
        super().__init__()
        self.config = config
        self.dim = config.dim
        max_heads = config.num_heads_max

        # 簇内 IJGP 层
        self.intra_ijgp = IntraClusterIJGP(
            config.dim, max_heads, edge_feat_dim, config.dropout
        )

        # 簇间 IJGP 层 (传入 edge_feat_dim 以支持共享变量特征 g(S_αβ))
        self.inter_ijgp = InterClusterIJGP(
            config.dim, config.dropout, edge_feat_dim=config.dim
        )

    def forward(self, x, edge_index, edge_attr, cluster_mask, num_heads,
                clusters=None, cluster_vars=None, cluster_cons=None,
                cluster_adj=None, factor_values=None, sat_scores=None):
        """
        Args:
            x:              [N, dim]            节点特征矩阵
            edge_index:     [2, E]              因子图边
            edge_attr:      [E, edge_feat_dim]  边特征
            cluster_mask:   [N]                 节点->簇ID 映射
            num_heads:      int                 当前活跃注意力头数
            clusters:       list[list]          每簇节点列表 (全局索引)
            cluster_vars:   list[list]          每簇变量列表 (全局索引)
            cluster_cons:   list[list]          每簇子句列表 (0-based)
            cluster_adj:    list[(i,j,shared)]  簇邻接关系
            factor_values:  [num_clauses] 或 None  因子函数值 
            sat_scores:     [num_clauses] 或 None  约束满足度 

        Returns:
            updated_x: [N, dim] 更新后的节点特征
        """
        # ===== 簇内 IJGP 消息传递 =====
        x = self.intra_ijgp(
            x, edge_index, edge_attr, cluster_mask,
            clusters, cluster_vars, cluster_cons, num_heads,
            factor_values=factor_values
        )

        # ===== 簇间 IJGP 消息传递 =====
        if cluster_adj is not None and len(cluster_adj) > 0:
            x = self.inter_ijgp(
                x, cluster_adj, cluster_vars, cluster_cons,
                sat_scores=sat_scores
            )

        return x

    def reset_counts(self):
        self.intra_ijgp.reset_counts()
        self.inter_ijgp.reset_counts()

    def get_counts(self):
        intra = self.intra_ijgp.get_counts()
        inter = self.inter_ijgp.get_counts()
        return {
            'intra_dot': intra['dot'],
            'intra_softmax': intra['softmax'],
            'inter_dot': inter['dot'],
            'total_dot': intra['dot'] + inter['dot'],
        }

    def set_recording(self, flag):
        """开关计数"""
        self.intra_ijgp._recording = flag
        self.inter_ijgp._recording = flag