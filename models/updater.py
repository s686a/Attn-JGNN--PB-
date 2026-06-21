import torch
import torch.nn as nn


class IJGPUpdater(nn.Module):
    """
    节点特征 GRU 更新

    h_v^{(t+1)} = GRU( m_v,  h_v^{(t)} )   变量: 输入 = 聚合消息, 隐藏 = 上一轮特征
    h_c^{(t+1)} = GRU( m_c,  h_c^{(t)} )   子句: 同上

    其中 m_v, m_c 是注意力层聚合的消息 (残差部分 ≈ updated - prev)。
    """

    def __init__(self, dim):
        super().__init__()
        self.gru_var = nn.GRUCell(dim, dim)
        self.gru_clause = nn.GRUCell(dim, dim)

    def forward(self, node_feats, prev_feats, num_vars):
        """
        参数:
            node_feats: [N, dim] 本轮注意力输出 (含残差 x + message)
            prev_feats: [N, dim] 上一轮节点特征
            num_vars:   int     变量节点数

        返回:
            updated: [N, dim] GRU 更新后的特征
        """
        # 消息 ≈ 残差部分 (注意力层输出 - 输入)
        var_msg = node_feats[:num_vars] - prev_feats[:num_vars]
        var_hidden = prev_feats[:num_vars]
        clause_msg = node_feats[num_vars:] - prev_feats[num_vars:]
        clause_hidden = prev_feats[num_vars:]

        # GRU: 输入=聚合消息, 隐藏状态=历史特征
        var_out = self.gru_var(var_msg, var_hidden)
        clause_out = self.gru_clause(clause_msg, clause_hidden)

        return torch.cat([var_out, clause_out], dim=0)