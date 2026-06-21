"""
系数感知的图注意力层 
此模块现已整合到 IntraClusterIJGP 中。
IntraClusterIJGP 实现了完整的:
  - 加性/拼接式注意力 
  - 因子值边际化 
  - 分段 softmax 
  - 可学习注意力头权重 

保留此文件作为参考实现和独立的 GAT 层使用。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoefficientAwareGATLayer(nn.Module):
    """
    系数感知的图注意力层，支持边特征（SAT 极性 / PB 系数+极性）和动态注意力头数。
    内部使用最大头数初始化，运行时可通过 heads 属性动态选择活跃头数。
    """
    def __init__(self, in_dim, out_dim, max_heads, edge_feat_dim, dropout=0.0):
        super().__init__()
        self.max_heads = max_heads
        self.heads = max_heads       # 当前活跃头数，可运行时修改
        self.out_dim = out_dim
        self.edge_feat_dim = edge_feat_dim
        self.dropout = dropout

        # 线性变换，输出维度固定为 max_heads * out_dim
        self.W = nn.Linear(in_dim, max_heads * out_dim, bias=False)
        # 注意力向量，输入为 2*out_dim + edge_feat_dim（每注意力头共享）
        self.att = nn.Linear(2 * out_dim + edge_feat_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_attr):
        """
        x: [N, in_dim]
        edge_index: [2, E]
        edge_attr: [E, edge_feat_dim]
        """
        N, E = x.size(0), edge_index.size(1)
        row, col = edge_index

        # 1. 线性变换并重塑为 [N, max_heads, out_dim]，只取活跃的 heads
        h = self.W(x).view(N, self.max_heads, self.out_dim)
        h = h[:, :self.heads, :]  # [N, active_heads, out_dim]

        # 2. 获取源节点和目标节点的特征 [E, active_heads, out_dim]
        h_row = h[row]
        h_col = h[col]

        # 3. 扩展边特征到每个活跃 head [E, active_heads, edge_feat_dim]
        edge_attr_exp = edge_attr.unsqueeze(1).expand(-1, self.heads, -1)

        # 4. 拼接特征计算注意力分数
        att_input = torch.cat([h_row, h_col, edge_attr_exp], dim=-1)  # [E, active_heads, 2*out_dim+edge_feat_dim]
        att_scores = self.leaky_relu(self.att(att_input).squeeze(-1))  # [E, active_heads]

        # 5. 按目标节点做 softmax 归一化
        max_vals = torch.zeros(N, self.heads, device=x.device)
        max_vals.scatter_reduce_(0, col.unsqueeze(-1).expand(-1, self.heads),
                                 att_scores, reduce='amax', include_self=False)
        exp_scores = torch.exp(att_scores - max_vals[col])
        sum_exp = torch.zeros(N, self.heads, device=x.device)
        sum_exp.scatter_add_(0, col.unsqueeze(-1).expand(-1, self.heads), exp_scores)
        att_weights = exp_scores / (sum_exp[col] + 1e-8)

        if self.dropout > 0 and self.training:
            att_weights = F.dropout(att_weights, p=self.dropout)

        # 6. 聚合消息
        msg = att_weights.unsqueeze(-1) * h_col

        out = torch.zeros(N, self.heads, self.out_dim, device=x.device)
        out.index_add_(0, row, msg)

        # 7. 拼接活跃 heads 的输出
        return out.view(N, self.heads * self.out_dim)