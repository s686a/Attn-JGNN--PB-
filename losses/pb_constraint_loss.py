import torch
import torch.nn as nn
from torch_scatter import scatter_sum


class PBConstraintLoss(nn.Module):
    def forward(self, var_probs, edge_index, edge_attr, bounds, num_vars):
        """
        PB 软约束损失

        var_probs: [num_vars] 每个变量为真的概率 ∈ [0, 1]
        edge_index: [2, E], row=变量节点, col=约束节点
        edge_attr: [E, 2], [:,0]=coeff, [:,1]=polarity (1正/-1负)
        bounds: [num_constraints] 每个约束的下界
        num_vars: int
        """
        coeffs = edge_attr[:, 0]
        polarities = edge_attr[:, 1]
        row, col = edge_index

        # 正确 PB 软语义:
        #   正文字 (polarity=1): coeff * p        (变量为真 → 贡献 coeff)
        #   负文字 (polarity=-1): coeff * (1-p)   (变量为假 → 贡献 coeff)
        pos_mask = (polarities > 0).float()
        neg_mask = (polarities < 0).float()
        weighted = coeffs * (var_probs[row] * pos_mask + (1 - var_probs[row]) * neg_mask)

        # 按约束聚合
        constr_sum = scatter_sum(weighted, col - num_vars, dim=0)

        # 约束为 sum >= bound，sigmoid 给出软满足度
        sat_scores = torch.sigmoid(constr_sum - bounds.to(var_probs.device))
        loss = -torch.log(sat_scores + 1e-8).mean()
        return loss