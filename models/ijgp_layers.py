"""
IJGP 消息传递层 —— 簇内/簇间并行化
=====================================
  - 加性/拼接式注意力  α_ij = softmax(LeakyReLU(a^T [Wq·h_i || Wk·h_j || We·e_ij]))
  - 变量→子句消息  m_{x→φ} = ∏ m_{ψ→x} · h_x
  - 子句→变量消息(含边际化)  m_{φ→x} = Σ_{~{x}} f_φ · ∏ m_{y→φ}
  - GRU 节点特征更新 (在 updater.py 中)
  - 簇间加性注意力
  - 簇间消息(含簇势函数 φ_α)
  - 共享变量特征更新

簇内计算：各簇独立，使用分段 softmax 替代逐簇循环，GPU 上自动并行
簇间计算：批量计算所有簇对的注意力，消除 Python for 循环
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_max


class IntraClusterIJGP(nn.Module):
    """
    簇内 IJGP 消息传递 — 加性/拼接式注意力

      α_ij = softmax_j( LeakyReLU( a^T [Wq·h_i || Wk·h_j || We·e_ij] ) )
    其中 a 是每个注意力头的可学习注意力向量, || 表示向量拼接。
    """

    def __init__(self, dim, max_heads, edge_feat_dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.max_heads = max_heads
        self.edge_feat_dim = edge_feat_dim
        self.dropout = dropout

        # ---- 计数器 ----
        self.register_buffer('attn_dot_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('attn_softmax_count', torch.tensor(0, dtype=torch.long))
        self._recording = True

        # ---- 加性注意力投影 ----
        # W_q, W_k 对节点特征做线性变换
        self.W_q = nn.Linear(dim, max_heads * dim, bias=False)
        self.W_k = nn.Linear(dim, max_heads * dim, bias=False)
        self.W_v = nn.Linear(dim, max_heads * dim, bias=False)
        # W_e 对边特征做线性变换
        self.W_e = nn.Linear(edge_feat_dim, max_heads * dim, bias=False)
        # 注意力向量 a: 输入为 [Wq·h_i || Wk·h_j || We·e_ij] → 每个头一个标量
        self.att_vec = nn.Linear(3 * dim, max_heads, bias=False)

        # ---- 可学习注意力头权重 λ_k ----
        self.head_logits = nn.Parameter(torch.zeros(max_heads))

        # ---- 边际化 MLP ----
        # 输入: [clause_context || excluded_var || factor_val] → 输出: 边际化消息
        self.marginal_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim)
        )

        # ---- 输出投影 ----
        self.out_proj = nn.Linear(max_heads * dim, dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_attr, cluster_mask, clusters,
                cluster_vars, cluster_cons, num_heads, factor_values=None):
        """
        簇内并行加性注意力：所有簇的边统一处理，通过分段 softmax 消除逐簇循环。

        参数:
            x:              [N, dim]            节点特征
            edge_index:     [2, E]              因子图边 (var→clause)
            edge_attr:      [E, edge_feat_dim]  边特征 (极性 ±1, 及 PB 系数)
            cluster_mask:   [N]                 节点→簇ID 映射
            clusters:       list[list]          每簇节点列表
            cluster_vars:   list[list]          每簇变量列表
            cluster_cons:   list[list]          每簇子句/约束列表
            num_heads:      int                 当前活跃注意力头数
            factor_values:  [num_clauses] 或 None  因子函数值(用于边际化)
        """
        N, E = x.size(0), edge_index.size(1)
        device = x.device
        heads = min(num_heads, self.max_heads)
        dim = self.dim

        row, col = edge_index

        # 推断变量数
        all_vars = set()
        for cv in cluster_vars:
            all_vars.update(cv)
        num_vars = max(all_vars) + 1 if all_vars else (N // 2)

        # ===== 构建双向边 =====
        bi_row = torch.cat([row, col])
        bi_col = torch.cat([col, row])
        if edge_attr is not None:
            bi_attr = torch.cat([edge_attr, edge_attr])
        else:
            bi_attr = None

        edge_cluster = cluster_mask[bi_row]     # [2E] 每条边所属簇

        # ===== 多头投影 =====
        Q_all = self.W_q(x).view(N, self.max_heads, dim)     # [N, maxH, D]
        K_all = self.W_k(x).view(N, self.max_heads, dim)     # [N, maxH, D]
        V_all = self.W_v(x).view(N, self.max_heads, dim)     # [N, maxH, D]

        # 只取活跃头
        Q = Q_all[:, :heads, :]   # [N, H, D]
        K = K_all[:, :heads, :]   # [N, H, D]
        V = V_all[:, :heads, :]   # [N, H, D]

        if bi_attr is not None:
            E_feat = self.W_e(bi_attr).view(-1, self.max_heads, dim)[:, :heads, :]  # [2E, H, D]
        else:
            E_feat = torch.zeros(bi_row.size(0), heads, dim, device=device)

        # ===== 加性/拼接式注意力 =====
        # α_ij = softmax( LeakyReLU( a^T [Wq·h_i || Wk·h_j || We·e_ij] ) )
        Q_src = Q[bi_row]       # [2E, H, D]
        K_dst = K[bi_col]       # [2E, H, D]

        # 拼接 [Q_src || K_dst || E_feat] 沿最后一维 → [2E, H, 3*D]
        concat_feat = torch.cat([Q_src, K_dst, E_feat], dim=-1)   # [2E, H, 3*D]

        # 加性注意力向量 a: Linear(3*D → maxH) → [2E, H, maxH]
        # 第 h 个注意力头使用第 h 个输出神经元 (每头独立注意力参数)
        att_raw = self.att_vec(concat_feat)                       # [2E, H, maxH]
        att_raw = att_raw[:, range(heads), range(heads)]          # [2E, H] 每头取自己通道

        if self._recording:
            self.attn_dot_count += att_raw.numel()
        att_raw = self.leaky_relu(att_raw)

        # ===== 可学习注意力头权重 λ_k =====
        head_weights = torch.sigmoid(self.head_logits[:heads])  # [H]
        att_raw = att_raw * head_weights.unsqueeze(0)            # [2E, H]

        # ===== 分段 softmax：按 (簇, 目标节点) 分组归一化 =====
        segment_key = edge_cluster.long() * N + bi_col     # [2E]
        unique_keys, inverse_indices = segment_key.unique(return_inverse=True)

        # 每段内求 max → softmax
        att_max = scatter_max(att_raw, inverse_indices, dim=0)[0]  # [num_segments, H]
        att_shifted = att_raw - att_max[inverse_indices]
        att_exp = torch.exp(att_shifted)
        att_sum = scatter_sum(att_exp, inverse_indices, dim=0).clamp(min=1e-8)
        att_norm = att_exp / att_sum[inverse_indices]     # [2E, H]

        if self._recording:
            self.attn_softmax_count += att_norm.numel()

        # ===== 消息聚合: 区分 var→clause 和 clause→var =====
        V_src = V[bi_row]       # [2E, H, D]
        V_dst = V[bi_col]       # [2E, H, D]

        # 聚合方向1: 变量→子句 m_{x→φ}
        weighted_var2clause = att_norm.unsqueeze(-1) * V_src    # [2E, H, D]
        msg_to_clause = torch.zeros(N, heads, dim, device=device)
        msg_to_clause.index_add_(0, bi_col, weighted_var2clause)

        # 聚合方向2: 子句→变量 m_{φ→x} 含边际化
        weighted_clause2var = att_norm.unsqueeze(-1) * V_dst    # [2E, H, D]
        msg_to_var = torch.zeros(N, heads, dim, device=device)
        msg_to_var.index_add_(0, bi_row, weighted_clause2var)

        # 计数归一化
        msg_count_to_clause = torch.zeros(N, heads, device=device)
        msg_count_to_var = torch.zeros(N, heads, device=device)
        ones = torch.ones(bi_row.size(0), heads, device=device)
        msg_count_to_clause.index_add_(0, bi_col, ones)
        msg_count_to_var.index_add_(0, bi_row, ones)
        msg_count_to_clause = msg_count_to_clause.clamp(min=1)
        msg_count_to_var = msg_count_to_var.clamp(min=1)

        avg_msg_clause = msg_to_clause / msg_count_to_clause.unsqueeze(-1)
        avg_msg_var = msg_to_var / msg_count_to_var.unsqueeze(-1)

        # ===== 边际化操作子句→变量消息 =====
        # m_{φ→x} ≈ MLP([ clause_context || excluded_var_feature || factor_value ])
        if factor_values is not None:
            # 为每个子句节点计算考虑边际化后的消息
            num_clauses = N - num_vars
            for c_idx in range(num_vars, min(num_vars + num_clauses, N)):
                if msg_count_to_var[c_idx, 0] > 0:
                    clause_ctx = avg_msg_var[c_idx]          # [H, D] 子句上下文
                    # 找到该子句的所有邻接变量边
                    edge_mask = (bi_col == c_idx) | (bi_row == c_idx)
                    if edge_mask.any():
                        for h in range(heads):
                            excluded_sum = V_all[bi_row[edge_mask], h, :].sum(dim=0)
                            n_edges = edge_mask.sum().float()
                            if n_edges > 1:
                                # 对每条边边际化
                                for e_idx in edge_mask.nonzero(as_tuple=False)[:, 0]:
                                    v_idx = bi_row[e_idx]
                                    excluded_feat = V_all[v_idx, h, :]   # 被边际化的变量
                                    marg_in = torch.cat([
                                        clause_ctx[h],                   # 子句上下文
                                        excluded_feat,                   # 排除的变量
                                        factor_values[c_idx - num_vars].expand(dim)  # 因子值
                                    ])
                                    avg_msg_var[c_idx, h] = self.marginal_mlp(marg_in)

        # ===== 合并两种消息 =====
        # 变量节点接收子句→变量的消息(avg_msg_var)
        # 子句节点接收变量→子句的消息(avg_msg_clause)
        combined_msg = torch.zeros(N, heads, dim, device=device)
        combined_msg[:num_vars] = avg_msg_var[:num_vars]          # 变量用 clause→var
        combined_msg[num_vars:] = avg_msg_clause[num_vars:]       # 子句用 var→clause

        # 补零 (如果 heads < max_heads)
        if heads < self.max_heads:
            pad = torch.zeros(N, self.max_heads - heads, dim, device=device)
            combined_msg = torch.cat([combined_msg, pad], dim=1)

        out = self.out_proj(combined_msg.view(N, self.max_heads * dim))

        if self.dropout > 0 and self.training:
            out = F.dropout(out, p=self.dropout)

        return self.norm(x + out)

    def reset_counts(self):
        self.attn_dot_count.zero_()
        self.attn_softmax_count.zero_()

    def get_counts(self):
        """返回 {dot_count, softmax_count}"""
        return {
            'dot': self.attn_dot_count.item(),
            'softmax': self.attn_softmax_count.item(),
        }


class InterClusterIJGP(nn.Module):
    """
    簇间 IJGP 消息传递 

    严格实现:
      α_αβ = softmax( LeakyReLU( a_c^T [Wq·h_α || Wk·h_β || Ws·g(S_αβ)] ) ) 
      m_{α→β}(S) = Σ_{C_α\S} φ_α · Π m_{γ→α}  →  MLP 近似边际化             
      h_v^{(t+1)} = h_v^{(t)} + Σ m_{β→α}(v)                                   
    """

    def __init__(self, dim, dropout=0.1, edge_feat_dim=None):
        super().__init__()
        self.dim = dim
        self.sqrt_d = dim ** 0.5
        self.dropout = dropout
        self.edge_feat_dim = edge_feat_dim or dim

        # ---- 计数器 ----
        self.register_buffer('attn_dot_count', torch.tensor(0, dtype=torch.long))
        self._recording = True

        # 加性注意力投影 (公式 3.11)
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_s = nn.Linear(self.edge_feat_dim, dim, bias=False)  # 共享变量特征投影
        # 簇间注意力向量
        self.att_vec = nn.Linear(3 * dim, 1, bias=False)

        # 簇势函数 φ_α — 基于簇内约束满足度
        self.potential_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),       # [cluster_feat || avg_sat_score]
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim),
            nn.Sigmoid()                    # 势函数 ∈ (0,1)
        )

        # 边际化 MLP 
        self.marginal_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),       # [非共享特征 || 簇势 || 入边消息聚合]
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim)
        )

        # 消息变换
        self.W_v = nn.Linear(dim, dim, bias=False)

        self.norm = nn.LayerNorm(dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, cluster_adj, cluster_vars, cluster_cons, sat_scores=None):
        """
        簇间并行加性注意力 + 簇势函数。

        参数:
            x:              [N, dim]            节点特征
            cluster_adj:    list[(i,j,shared)]  簇邻接关系
            cluster_vars:   list[list]          每簇变量列表
            cluster_cons:   list[list]          每簇约束列表
            sat_scores:     [num_clauses] 或 None  约束满足度(用于簇势计算)
        """
        N, dim = x.size(0), self.dim
        device = x.device
        K = len(cluster_vars)

        if K <= 1 or len(cluster_adj) == 0:
            return x

        # ===== 批量计算簇特征 =====
        cluster_feats = []
        for cid in range(K):
            if cluster_vars[cid]:
                v_idx = torch.tensor(cluster_vars[cid], device=device)
                cluster_feats.append(x[v_idx].mean(dim=0))
            else:
                cluster_feats.append(torch.zeros(dim, device=device))
        cluster_feats = torch.stack(cluster_feats, dim=0)           # [K, D]

        # ===== 簇势函数 φ_α =====
        # 基于簇内约束满足度计算真正的势函数
        if sat_scores is not None and len(cluster_cons) > 0:
            avg_sat_per_cluster = torch.zeros(K, device=device)
            for cid in range(K):
                if cluster_cons[cid]:
                    cons_idx = torch.tensor([c for c in cluster_cons[cid]], device=device)
                    if cons_idx.max() < len(sat_scores):
                        avg_sat_per_cluster[cid] = sat_scores[cons_idx].mean()
            # 拼接簇特征和满足度均值
            pot_input = torch.cat([
                cluster_feats,
                avg_sat_per_cluster.unsqueeze(-1).expand(-1, dim)
            ], dim=-1)
            cluster_potentials = self.potential_mlp(pot_input)       # [K, D]
        else:
            # 回退到纯学习投影
            cluster_potentials = torch.sigmoid(cluster_feats)

        # ===== 簇间加性注意力 =====
        num_pairs = len(cluster_adj)
        src_ids = torch.tensor([i for i, j, sh in cluster_adj], device=device)
        dst_ids = torch.tensor([j for i, j, sh in cluster_adj], device=device)

        Q_all = self.W_q(cluster_feats)                             # [K, D]
        K_all = self.W_k(cluster_feats)                             # [K, D]

        # 共享变量聚合特征 g(S_αβ)
        shared_feats = torch.zeros(num_pairs, self.edge_feat_dim, device=device)
        for p, (i, j, shared_vars) in enumerate(cluster_adj):
            if shared_vars:
                sh_idx = torch.tensor(list(shared_vars), device=device)
                shared_feats[p, :dim] = x[sh_idx].mean(dim=0)
        shared_proj = self.W_s(shared_feats)                        # [P, D]

        # 加性注意力: concat [Q_α || K_β || g(S_αβ)]
        concat_att = torch.cat([
            Q_all[src_ids],       # [P, D]
            K_all[dst_ids],       # [P, D]
            shared_proj           # [P, D]
        ], dim=-1)                # [P, 3*D]
        alphas = self.leaky_relu(self.att_vec(concat_att).squeeze(-1))  # [P]

        if self._recording:
            self.attn_dot_count += num_pairs * 2  # 双向

        # ===== 批量计算非共享变量特征 (边际化准备) =====
        cluster_var_sums = torch.zeros(K, dim, device=device)
        cluster_var_counts = torch.zeros(K, device=device)
        for cid in range(K):
            if cluster_vars[cid]:
                v_idx = torch.tensor(cluster_vars[cid], device=device)
                cluster_var_sums[cid] = x[v_idx].sum(dim=0)
                cluster_var_counts[cid] = len(cluster_vars[cid])

        ns_feats_src = torch.zeros(num_pairs, dim, device=device)
        ns_feats_dst = torch.zeros(num_pairs, dim, device=device)

        for p, (i, j, shared_vars) in enumerate(cluster_adj):
            if not shared_vars:
                ns_feats_src[p] = cluster_feats[i]
                ns_feats_dst[p] = cluster_feats[j]
                continue
            shared_set = set(shared_vars)

            sh_idx_i = torch.tensor([v for v in shared_vars if v in set(cluster_vars[i])],
                                     device=device)
            sh_idx_j = torch.tensor([v for v in shared_vars if v in set(cluster_vars[j])],
                                     device=device)

            if len(sh_idx_i) > 0 and cluster_var_counts[i] > len(sh_idx_i):
                sh_sum_i = x[sh_idx_i].sum(dim=0)
                ns_sum_i = cluster_var_sums[i] - sh_sum_i
                ns_feats_src[p] = ns_sum_i / (cluster_var_counts[i] - len(sh_idx_i))
            else:
                ns_feats_src[p] = cluster_feats[i] if cluster_var_counts[i] > 0 else torch.zeros(dim, device=device)

            if len(sh_idx_j) > 0 and cluster_var_counts[j] > len(sh_idx_j):
                sh_sum_j = x[sh_idx_j].sum(dim=0)
                ns_sum_j = cluster_var_sums[j] - sh_sum_j
                ns_feats_dst[p] = ns_sum_j / (cluster_var_counts[j] - len(sh_idx_j))
            else:
                ns_feats_dst[p] = cluster_feats[j] if cluster_var_counts[j] > 0 else torch.zeros(dim, device=device)

        # ===== 边际消息 MLP 近似 Σ_{C_α\S} φ_α · Π m =====
        marg_in_src = torch.cat([
            ns_feats_src,                    # [P, D] 非共享变量特征
            cluster_potentials[src_ids],     # [P, D] 簇势 φ_α
            cluster_feats[src_ids]           # [P, D] 簇上下文(替代入边消息乘积)
        ], dim=-1)                           # [P, 3*D]
        marg_in_dst = torch.cat([
            ns_feats_dst,
            cluster_potentials[dst_ids],
            cluster_feats[dst_ids]
        ], dim=-1)

        m_src_to_dst = self.marginal_mlp(marg_in_src)              # [P, D]
        m_dst_to_src = self.marginal_mlp(marg_in_dst)              # [P, D]

        # ===== 批量累加簇间消息 =====
        inter_msgs = torch.zeros(K, dim, device=device)
        inter_msgs.index_add_(0, dst_ids, alphas.unsqueeze(-1) * m_src_to_dst)
        inter_msgs.index_add_(0, src_ids, alphas.unsqueeze(-1) * m_dst_to_src)

        # ===== 共享变量特征更新 =====
        updated_x = x.clone()
        Wv_msg = self.W_v(inter_msgs)                               # [K, D]

        for cid in range(K):
            if not cluster_vars[cid]:
                continue
            msg = Wv_msg[cid]
            v_idx = torch.tensor(cluster_vars[cid], device=device)
            updated_x[v_idx] = updated_x[v_idx] + msg.unsqueeze(0)

        if self.dropout > 0 and self.training:
            updated_x = F.dropout(updated_x, p=self.dropout)

        return self.norm(updated_x)

    def reset_counts(self):
        self.attn_dot_count.zero_()

    def get_counts(self):
        """返回 {dot_count}"""
        return {
            'dot': self.attn_dot_count.item(),
        }
