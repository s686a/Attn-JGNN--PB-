import torch
import torch.nn as nn
from torch_scatter import scatter_sum

class SATConstraintLoss(nn.Module):
    def forward(self, var_probs, edge_index, edge_attr, num_vars):
        # edge_attr: [E, 1] polarity
        polarities = edge_attr[:, 0]
        coeffs = torch.ones_like(polarities)
        row, col = edge_index
        weighted = coeffs * (2 * var_probs[row] - 1) * polarities
        clause_sum = scatter_sum(weighted, col - num_vars, dim=0)
        sat_scores = torch.sigmoid(clause_sum - 1.0)
        loss = -torch.log(sat_scores + 1e-8).mean()
        return loss
