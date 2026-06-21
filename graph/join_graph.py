from .tree_decomposition import adaptive_tree_decomposition
import torch


def build_join_graph(data, flow_cutter_path, clauses, factor_graph_edges):
    """
    构建连接图，包含簇划分、共享变量等信息。
    data: PyG Data 对象（已有 edge_index, edge_attr, num_vars, num_clauses）
    clauses: CNF 子句列表 (list of list of int) 或 PB 约束列表 (list of tuples)
    factor_graph_edges: list of (u,v) 因子图边
    返回: data 对象添加 clusters, cluster_vars, cluster_cons, cluster_adj, var_to_clusters, cluster_mask
    """
    num_vars = data.num_vars
    num_clauses = data.num_clauses

    # 自适应树分解（支持 CNF 和 PB 两种 clause 格式）
    clusters, actual_width = adaptive_tree_decomposition(
        num_vars, num_clauses, clauses, factor_graph_edges, flow_cutter_path
    )

    # 记录每个簇的变量和子句/约束
    cluster_vars = []
    cluster_cons = []
    for nodes in clusters:
        vars_in = [n for n in nodes if n < num_vars]
        cons_in = [n - num_vars for n in nodes if n >= num_vars]
        cluster_vars.append(vars_in)
        cluster_cons.append(cons_in)

    # 构建簇间邻接及共享变量
    K = len(clusters)
    cluster_adj = []   # (i, j, shared_vars_set)
    for i in range(K):
        vars_i = set(cluster_vars[i])
        for j in range(i+1, K):
            shared = vars_i.intersection(set(cluster_vars[j]))
            if shared:
                cluster_adj.append((i, j, shared))

    # 变量到簇的映射
    var_to_clusters = [[] for _ in range(num_vars)]
    for cid, vars_list in enumerate(cluster_vars):
        for v in vars_list:
            var_to_clusters[v].append(cid)

    # 簇掩码（为每个节点标记其所属的簇 ID）
    total_nodes = num_vars + num_clauses
    cluster_mask = torch.full((total_nodes,), -1, dtype=torch.long)
    for cid, nodes in enumerate(clusters):
        for n in nodes:
            cluster_mask[n] = cid

    # 将簇信息存入 data 对象
    data.clusters = clusters
    data.cluster_vars = cluster_vars
    data.cluster_cons = cluster_cons
    data.cluster_adj = cluster_adj
    data.var_to_clusters = var_to_clusters
    data.cluster_mask = cluster_mask
    data.used_treewidth = actual_width
    return data


def build_join_graph_adaptive(data, flow_cutter_path, task='sat'):
    """
    自适应构建连接图的便捷接口。
    从 data 对象中提取 clauses 和边信息，调用 build_join_graph。
    """
    num_vars = data.num_vars
    num_clauses = data.num_clauses

    # 提取因子图边
    if hasattr(data, 'edge_index'):
        factor_graph_edges = list(data.edge_index.t().tolist())
    else:
        factor_graph_edges = []

    # 提取 clauses/constraints
    if hasattr(data, 'clauses'):
        clauses = data.clauses
    else:
        # 从 edge_index 推断（仅 SAT 场景）
        clauses = []
        if hasattr(data, 'edge_index') and hasattr(data, 'edge_attr'):
            # 按约束分组重建
            var_to_constraints = {}
            for e_idx in range(data.edge_index.size(1)):
                var = data.edge_index[0, e_idx].item()
                con = data.edge_index[1, e_idx].item() - num_vars
                if con not in var_to_constraints:
                    var_to_constraints[con] = []
                polarity = data.edge_attr[e_idx, 0].item()
                lit = (var + 1) if polarity > 0 else -(var + 1)
                var_to_constraints[con].append(lit)
            clauses = [var_to_constraints.get(c, []) for c in range(num_clauses)]

    return build_join_graph(data, flow_cutter_path, clauses, factor_graph_edges)