"""
动态注意力头分配器 (论文 3.7.1 节 + 算法 3)

实现两层机制:
  1. 按训练步数 t 全局增长: H(t) = H_init + ⌊t / T_step⌋ (公式 3.1)
  2. 按簇复杂度得分 S_α 为每个簇独立分配头数 (算法 3)
     S_α = w_size·|C_α| + w_density·ρ_α + w_connectivity·σ_α  (公式 3.21)
"""
import math


class DynamicHeadAllocator:
    """
    动态注意力头分配器。

    在训练过程中:
      - 全局头数 H_global 随 epoch/step 线性增长 (公式 3.1)
      - 各簇根据复杂度得分 S_α 按比例分配 H_global 个头 (算法 3)
      - 剩余头 (因取整) 分配给复杂度最高的簇
    """

    def __init__(self, config):
        self.init_heads = config.num_heads_init
        self.max_heads = config.num_heads_max
        self.step = config.head_increase_step          # 每 N 步增加 1 个头
        self.weight_size = getattr(config, 'w_cluster_size', 0.5)
        self.weight_density = getattr(config, 'w_cluster_density', 0.3)
        self.weight_connect = getattr(config, 'w_cluster_connect', 0.2)

    def get_global_heads(self, epoch):
        """公式 (3.1): 全局头数随训练步数增长"""
        return min(self.init_heads + epoch // self.step, self.max_heads)

    def compute_cluster_scores(self, clusters, cluster_vars, cluster_cons, edge_index):
        """
        公式 (3.21): 计算每个簇的复杂度得分 S_α。

        参数:
            clusters:       list[list]  每簇节点列表
            cluster_vars:   list[list]  每簇变量列表
            cluster_cons:   list[list]  每簇约束列表
            edge_index:     [2, E]      因子图边

        返回:
            scores: list[float] 各簇复杂度得分
        """
        K = len(clusters)
        scores = []
        for cid in range(K):
            # 簇规模 |C_α|
            size = len(clusters[cid])

            # 簇内约束密度 ρ_α
            nodes_set = set(clusters[cid])
            edge_count = 0
            for u, v in edge_index.t().tolist():
                if u in nodes_set and v in nodes_set:
                    edge_count += 1
            density = edge_count / (size * (size - 1) + 1e-8) if size > 1 else 0.0

            # 簇间关联度 σ_α
            shared = 0
            for other_cid in range(K):
                if other_cid != cid:
                    shared += len(set(cluster_vars[cid]) & set(cluster_vars[other_cid]))

            # 综合得分 (公式 3.21)
            S = (self.weight_size * size +
                 self.weight_density * density * 10 +
                 self.weight_connect * shared)
            scores.append(S)
        return scores

    def allocate_per_cluster(self, global_heads, cluster_scores):
        """
        算法 3: 按簇复杂度得分占比分配头数。

        参数:
            global_heads:   int           当前全局头数
            cluster_scores: list[float]   各簇复杂度得分

        返回:
            allocation: list[int] 各簇分配的头数
        """
        K = len(cluster_scores)
        if K == 0:
            return []

        total_score = sum(cluster_scores)
        if total_score == 0:
            # 均分
            base = global_heads // K
            remainder = global_heads % K
            return [base + (1 if i < remainder else 0) for i in range(K)]

        # 按比例分配 (向下取整)
        allocation = []
        for score in cluster_scores:
            heads = int(global_heads * score / total_score)
            heads = max(1, heads)  # 每个簇至少 1 个头
            allocation.append(heads)

        # 剩余头分配给复杂度最高的簇 (算法 3 第 8-9 行)
        total_allocated = sum(allocation)
        remaining = global_heads - total_allocated
        while remaining > 0:
            best_idx = cluster_scores.index(max(cluster_scores))
            allocation[best_idx] += 1
            remaining -= 1

        # 确保不超过 max_heads
        allocation = [min(h, self.max_heads) for h in allocation]

        return allocation
