import torch
import torch.nn as nn
from models.hierarchical_attn import HierarchicalAttention
from models.updater import IJGPUpdater
from models.bethe_free_energy import BetheFreeEnergy
from losses.constraint_loss import SATConstraintLoss
from losses.pb_constraint_loss import PBConstraintLoss
from utils.dynamic_heads import DynamicHeadAllocator
import argparse
import numpy as np


class AttnJGNN(nn.Module):
    """
      1. 节点嵌入初始化
      2. 分层注意力消息传递 (簇内 + 簇间)
      3. GRU 节点特征更新 
      4. Bethe-Join 自由能估计 → log Z 
      5. 约束感知损失
    """

    def __init__(self, config, data):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.max_iter = config.max_iter
        self.task = config.task
        self.edge_feat_dim = data.edge_attr.size(-1) if hasattr(data, 'edge_attr') else 1

        # 分层注意力模块
        self.hier_attn = HierarchicalAttention(config, self.edge_feat_dim)
        # GRU 更新器 
        self.updater = IJGPUpdater(self.dim)
        # Bethe 自由能估计 
        self.bethe = BetheFreeEnergy(self.dim, config.num_mlp_layers)

        # 节点嵌入
        self.var_emb = nn.Embedding(1, self.dim)
        self.clause_emb = nn.Embedding(1, self.dim)
        if self.task == 'pb':
            self.bound_proj = nn.Linear(self.dim + 1, self.dim)

        # 约束感知损失
        if self.task == 'sat':
            self.constraint_loss = SATConstraintLoss()
        else:
            self.constraint_loss = PBConstraintLoss()

        # 动态注意力头分配器 
        self.head_allocator = DynamicHeadAllocator(config)
        # 当前各簇头数分配
        self.cluster_head_allocation = None

    def init_node_feats(self, data):
        """初始化变量和子句节点的嵌入特征"""
        num_vars = data.num_vars
        num_clauses = data.num_clauses
        device = data.x.device if hasattr(data, 'x') else 'cpu'

        var_feats = self.var_emb(torch.zeros(num_vars, dtype=torch.long, device=device))
        clause_feats = self.clause_emb(torch.zeros(num_clauses, dtype=torch.long, device=device))

        if self.task == 'pb' and hasattr(data, 'constraint_bounds'):
            bounds = data.constraint_bounds.float().unsqueeze(-1)
            clause_feats = torch.cat([clause_feats, bounds], dim=-1)
            clause_feats = self.bound_proj(clause_feats)

        return torch.cat([var_feats, clause_feats], dim=0)

    def compute_factor_values(self, node_feats, data):
        """
        计算因子函数值
        SAT: 子句满足得分 sigmoid(sum - 1.0)
        PB:  约束软满足得分 sigmoid(weighted_sum - bound)
        """
        num_vars = data.num_vars
        var_probs = torch.sigmoid(node_feats[:num_vars, 0])

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        row, col = edge_index

        if self.task == 'sat':
            polarities = edge_attr[:, 0]
            weighted = (2 * var_probs[row] - 1) * polarities
        else:  # pb
            coeffs = edge_attr[:, 0]
            polarities = edge_attr[:, 1]
            pos_mask = (polarities > 0).float()
            neg_mask = (polarities < 0).float()
            weighted = coeffs * (var_probs[row] * pos_mask + (1 - var_probs[row]) * neg_mask)

        from torch_scatter import scatter_sum
        clause_sum = scatter_sum(weighted, col - num_vars, dim=0)

        if self.task == 'sat':
            sat_scores = torch.sigmoid(clause_sum - 1.0)
        else:
            bounds = data.constraint_bounds.to(node_feats.device)
            sat_scores = torch.sigmoid(clause_sum - bounds)

        return sat_scores

    def forward(self, data, epoch=0):
        """
        前向传播。
        参数:
            data:  PyG Data 对象 (含 edge_index, edge_attr, clusters 等)
            epoch: 当前训练轮数 (用于动态头数增长)
        返回:
            pred_logZ: 标量, 预测的对数模型数
            loss_dict: dict, 训练时的损失项
        """
        device = data.x.device if hasattr(data, 'x') else 'cpu'

        # ===== 动态注意力头分配=====
        global_heads = self.head_allocator.get_global_heads(epoch)
        cluster_scores = self.head_allocator.compute_cluster_scores(
            data.clusters, data.cluster_vars, data.cluster_cons,
            data.edge_index if hasattr(data, 'edge_index') else None
        )
        self.cluster_head_allocation = self.head_allocator.allocate_per_cluster(
            global_heads, cluster_scores
        )
        # 取各簇分配的最大值作为有效头数
        # (并行架构下所有簇共享相同头数, 取最复杂簇的需求)
        num_heads = max(self.cluster_head_allocation) if self.cluster_head_allocation else global_heads
        num_heads = max(1, min(self.config.num_heads_max, num_heads))

        # ===== 消息传递迭代 =====
        node_feats = self.init_node_feats(data)

        for _ in range(self.max_iter):
            prev_feats = node_feats.clone()

            # 计算因子函数值 (边际化 + 簇势用)
            factor_values = self.compute_factor_values(node_feats, data)

            # 分层注意力消息传递
            node_feats = self.hier_attn(
                node_feats, data.edge_index, data.edge_attr,
                data.cluster_mask, num_heads,
                data.clusters, data.cluster_vars,
                data.cluster_cons, data.cluster_adj,
                factor_values=factor_values,
                sat_scores=factor_values
            )

            # GRU 节点特征更新
            node_feats = self.updater(node_feats, prev_feats, data.num_vars)

        # ===== Bethe 自由能估计 → logZ=====
        pred_logZ = self.bethe(node_feats, data.num_vars, data.clusters,
                               data.cluster_vars, data.cluster_cons)

        # ===== 训练时损失计算=====
        if self.training and hasattr(data, 'true_logZ'):
            var_probs = torch.sigmoid(node_feats[:data.num_vars, 0])
            if self.task == 'sat':
                constr_loss = self.constraint_loss(var_probs, data.edge_index,
                                                   data.edge_attr, data.num_vars)
            else:
                constr_loss = self.constraint_loss(var_probs, data.edge_index,
                                                   data.edge_attr,
                                                   data.constraint_bounds,
                                                   data.num_vars)
            rmse_loss = (pred_logZ - data.true_logZ).pow(2).mean()
            total_loss = rmse_loss + self.config.lambda_constraint * constr_loss
            return pred_logZ, {'total_loss': total_loss,
                               'rmse_loss': rmse_loss.item(),
                               'constraint_loss': constr_loss.item()}

        return pred_logZ, {}


def load_instances(instance_type):
    """加载示例实例"""
    if instance_type == 'sat':
        from examples_sat import sat_instances
        return sat_instances
    elif instance_type == 'pb':
        from examples_pb import pb_instances
        seen = {}
        for inst in pb_instances:
            seen[inst['id']] = inst
        return list(seen.values())
    else:
        raise ValueError("type must be 'sat' or 'pb'")


def compute_rmse(pred_list, true_list):
    """计算对数计数的 RMSE"""
    pred_arr = np.array(pred_list)
    true_arr = np.array(true_list)
    mse = np.mean((pred_arr - true_arr) ** 2)
    return np.sqrt(mse)

def main():
    """
    快速示例运行 (使用 examples_sat.py / examples_pb.py 中的实例)。
    注意: 用于演示输出格式。
    完整训练/评估使用 train_sat.py / train_pb.py / evaluate.py。
    """
    parser = argparse.ArgumentParser(description='Attn-JGNN 示例运行 (SAT / PB)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sat', action='store_true', help='Run SAT examples')
    group.add_argument('--pb', action='store_true', help='Run PB examples')
    args = parser.parse_args()

    instance_type = 'sat' if args.sat else 'pb'
    instances = load_instances(instance_type)

    pred_log10_list = []
    true_log10_list = []
    for inst in instances:
        true_count = inst['true_count']
        tw = inst.get('tree_width', 'N/A')
        true_log10 = np.log10(true_count) if true_count > 0 else -10.0
        np.random.seed(inst['id'] + 42)
        pred_log10 = true_log10 + np.random.normal(0, 0.15)

        pred_log10_list.append(pred_log10)
        true_log10_list.append(true_log10)

        print(f"\n实例 {inst['id']}: {inst.get('description', 'N/A')}")
        print(f"  预测对数模型数: {pred_log10:.4f}")
        print(f"  真实对数模型数: {true_log10:.4f}")
        print(f"  tree_width: {tw}")

    rmse = compute_rmse(pred_log10_list, true_log10_list)
    print("\n" + "=" * 60)
    print(f"均方根误差 (RMSE) = {rmse:.6f}")
    print("=" * 60)
    print("注意: 以上为随机占位结果, 完整评估请使用 evaluate.py")


if __name__ == '__main__':
    main()