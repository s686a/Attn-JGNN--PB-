import torch
from torch_geometric.data import Data

def build_factor_graph(num_vars, clauses_or_constraints, task='sat'):
    """
    构建因子图（变量节点 + 子句/约束节点）
    返回 PyG Data 对象，包含 edge_index 和 edge_attr
    对于 #SAT: edge_attr 为 [polarity]
    对于 PB: edge_attr 为 [coeff, polarity]
    """
    if task == 'sat':
        num_clauses = len(clauses_or_constraints)
        edge_index = []
        edge_attr = []
        for cid, clause in enumerate(clauses_or_constraints):
            for lit in clause:
                var = abs(lit) - 1
                polarity = 1 if lit > 0 else -1
                edge_index.append([var, num_vars + cid])
                edge_attr.append([polarity])
        data = Data(edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                   edge_attr=torch.tensor(edge_attr, dtype=torch.float))
        data.num_vars = num_vars
        data.num_clauses = num_clauses
    else:  # pb
        num_constraints = len(clauses_or_constraints)
        edge_index = []
        edge_attr = []
        bounds = []
        for cid, (coeffs, lits, op, bound) in enumerate(clauses_or_constraints):
            bounds.append(bound)
            for coeff, lit in zip(coeffs, lits):
                var = abs(lit) - 1
                polarity = 1 if lit > 0 else -1
                edge_index.append([var, num_vars + cid])
                edge_attr.append([float(coeff), float(polarity)])
        data = Data(edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                   edge_attr=torch.tensor(edge_attr, dtype=torch.float))
        data.num_vars = num_vars
        data.num_clauses = num_constraints
        data.clauses = clauses_or_constraints
        data.constraint_bounds = torch.tensor(bounds, dtype=torch.float)

    data.x = torch.zeros(data.num_vars + data.num_clauses, 1)
    return data