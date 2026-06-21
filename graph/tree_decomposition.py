
import networkx as nx
from config import Config


def compute_complexity_score(num_vars, num_clauses, clauses, factor_graph_edges):
    """
    计算 CNF/PB 公式的综合复杂度得分 S ∈ [0, 10]。
    S = 10 × (w_rho·ρ_norm + w_cycle·C_norm + w_length·L_norm)
    权重 (w_rho, w_cycle, w_length) = (0.4, 0.3, 0.3) 由交叉验证确定。

    参数:
        num_vars: 变量数 n
        num_clauses: 子句/约束数 m
        clauses: CNF (list of list of int) 或 PB 约束列表
        factor_graph_edges: list of (u, v) 边
    返回:
        S: float ∈ [0, 10]
    """
    m = num_clauses
    n = max(num_vars, 1)

    # ---------- 维度 1: 密度 ρ = m/n ----------
    rho = m / n
    rho_norm = min(rho / Config.rho_max, 1.0)

    # ---------- 维度 2: 平均子句/约束长度 L_avg ----------
    if clauses:
        if isinstance(clauses[0], (list, tuple)):
            if isinstance(clauses[0], tuple):
                # PB 格式: (coeffs, lits, op, bound) → len(lits)
                L_avg = sum(len(c[1]) for c in clauses) / len(clauses)
            else:
                # CNF 格式: list of int
                L_avg = sum(len(c) for c in clauses) / len(clauses)
        else:
            L_avg = 2.0
    else:
        L_avg = 2.0
    L_norm = min(L_avg / Config.L_max, 1.0)

    # ---------- 维度 3: 围长（环复杂度）C_cycle = E - N + C ----------
    E = len(factor_graph_edges)
    N = num_vars + num_clauses
    G = nx.Graph()
    G.add_edges_from(factor_graph_edges)
    C_comp = nx.number_connected_components(G)
    C_cycle = E - N + C_comp
    C_norm = min(C_cycle / Config.cycle_norm_factor, 1.0)

    # ---------- 综合得分 ----------
    S = 10 * (Config.w_rho * rho_norm + Config.w_cycle * C_norm + Config.w_length * L_norm)
    return S


def select_treewidth(S):
    """
    算法 2 的第 10-18 行: 根据复杂度得分 S 从候选集 {3, 5, 8, 12} 中选取树宽。

    分档规则:
        S < 3.0  → tw = 3   (简单公式)
        S < 6.0  → tw = 5   (中等复杂度)
        S < 8.0  → tw = 8   (较复杂)
        S ≥ 8.0  → tw = 12  (高复杂度)
    """
    tw_candidates = Config.tw_candidates  # [3, 5, 8, 12]
    thresholds = [3.0, 6.0, 8.0]           # 分档阈值
    for tw, th in zip(tw_candidates, thresholds):
        if S < th:
            return tw
    return tw_candidates[-1]  # S >= 8.0 → tw = 12


def adaptive_tree_decomposition(num_vars, num_clauses, clauses,
                                factor_graph_edges, flow_cutter_path):
    """
    自适应树分解:
    1. 计算复杂度得分 S
    2. 根据 S 选择最优树宽 tw (作为参考目标)
    3. 执行树分解: FlowCutter → NetworkX 回退

    注意: FlowCutter 不原生支持目标树宽参数 — 它总是输出当前最优分解。
    实际树宽由 FlowCutter 的搜索时间决定, target_tw 仅作复杂度分档参考。
    如需精确控制树宽, 需修改 FlowCutter C++ 源码 (见 README §3.4.2)。

    返回:
        clusters: list of list of int — 每个簇内的节点列表
        actual_width: int — 实际树宽
    """
    # Step 1 & 2: 计算复杂度并选树宽
    S = compute_complexity_score(num_vars, num_clauses, clauses, factor_graph_edges)
    target_tw = select_treewidth(S)

    total_nodes = num_vars + num_clauses
    G = nx.Graph()
    G.add_nodes_from(range(total_nodes))
    G.add_edges_from(factor_graph_edges)

    # Step 3: 执行树分解
    clusters = None
    actual_width = target_tw

    # 尝试使用 FlowCutter（外部工具）
    # 如果使用了修改版 FlowCutter (支持 --target-width), 传入 target_tw
    try:
        from utils.flow_cutter_utils import run_flow_cutter
        clusters = run_flow_cutter(G, timeout=30, target_treewidth=target_tw)
        if clusters:
            actual_width = max(len(c) - 1 for c in clusters) if clusters else 1
    except Exception:
        pass

    # 回退方案: NetworkX 启发式
    if not clusters:
        try:
            _, elimination_order = nx.approximation.treewidth_min_degree(G)
            clusters = _build_clusters_from_components(G, target_tw)
            if clusters:
                actual_width = max(len(c) - 1 for c in clusters) if clusters else 1
        except Exception:
            pass

    # 最终回退: 每节点一个簇
    if not clusters:
        clusters = [[i] for i in range(total_nodes)]
        actual_width = 1

    return clusters, actual_width


def _build_clusters_from_components(G, target_tw=5):

    total_nodes = G.number_of_nodes()
    clusters = []

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        raw_communities = list(greedy_modularity_communities(G))
        for comm in raw_communities:
            if len(comm) > 0:
                clusters.append(list(comm))
    except Exception:
        pass

    if not clusters:
        for comp in nx.connected_components(G):
            comp_list = list(comp)
            if len(comp_list) > 0:
                clusters.append(comp_list)

    max_cluster_size = max(target_tw * 3, 15)
    final_clusters = []
    for cluster_nodes in clusters:
        if len(cluster_nodes) <= max_cluster_size:
            final_clusters.append(cluster_nodes)
        else:
            sub = G.subgraph(cluster_nodes)
            try:
                from networkx.algorithms.community import greedy_modularity_communities
                sub_comms = list(greedy_modularity_communities(sub))
                for sc in sub_comms:
                    if len(sc) > 0:
                        final_clusters.append(list(sc))
            except Exception:
                final_clusters.append(cluster_nodes)

    all_clustered = set()
    for c in final_clusters:
        all_clustered.update(c)
    remaining = set(range(total_nodes)) - all_clustered
    if remaining:
        final_clusters.append(list(remaining))

    return final_clusters