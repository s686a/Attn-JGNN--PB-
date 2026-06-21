"""
PB→CNF 转换
=============================
实现两种编码方法:
  1. Warners 编码 (加法器网络/Totalizer) 
  2. BDD 编码 (二元决策图 → Tseitin) 

    from data.pb_to_cnf import warners_encode, bdd_encode
    num_vars_new, cnf_clauses = warners_encode(pb_constraints, num_vars)
    num_vars_new, cnf_clauses = bdd_encode(pb_constraints, num_vars)
"""


def _sequential_counter_encode(vars_list, bound, aux_start):
    """
    顺序计数器 (Sinz 2005): 对 n 个变量中至多 bound 个为真进行编码。
    返回: (clauses, next_aux)
    """
    n = len(vars_list)
    clauses = []
    aux = aux_start
    if n == 0:
        return clauses, aux
    if bound <= 0:
        # 至多0个为真 → 所有变量必须为假
        for v in vars_list:
            clauses.append([-v])
        return clauses, aux

    # 寄存器 R[i][j]: 前 i 个输入中至少 j 个为真 (1<=i<=n, 1<=j<=bound)
    R = [[0] * (bound + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, bound + 1):
            R[i][j] = aux
            aux += 1

    # Sinz 编码子句
    for i in range(1, n + 1):
        x = vars_list[i - 1]
        clauses.append([-x, R[i][1]])                          # (1)
        if i <= n and bound < n and i > bound:
            clauses.append([-x, -R[i - 1][bound]])             # (5)

    for i in range(2, n + 1):
        clauses.append([-R[i - 1][1], R[i][1]])                # (2)
        for j in range(2, bound + 1):
            clauses.append([-vars_list[i - 1], -R[i - 1][j - 1], R[i][j]])  # (3)
            clauses.append([-R[i - 1][j], R[i][j]])            # (4)

    return clauses, aux


def _pb_to_at_least_k(vars_list, coeffs, k, aux_start):
    """将加权约束 Σ coeffs·x_i >= k 通过变量复制展开为基数约束。"""
    clauses = []
    aux = aux_start
    expanded = []

    for var, coeff in zip(vars_list, coeffs):
        c = int(round(coeff))
        if c <= 0:
            continue
        if c == 1:
            expanded.append(var)
        else:
            copies = []
            for _ in range(c):
                copies.append(aux)
                aux += 1
            for cp in copies:
                clauses.append([-var, cp])
                clauses.append([var, -cp])
            expanded.extend(copies)

    if len(expanded) < k:
        clauses.append([aux, -aux])
        return clauses, aux + 1

    n = len(expanded)
    complement = []
    for v in expanded:
        neg_v = aux
        aux += 1
        clauses.append([v, neg_v])
        clauses.append([-v, -neg_v])
        complement.append(neg_v)

    max_true = n - k
    if max_true < 0:
        clauses.append([aux, -aux])
        return clauses, aux + 1
    if max_true == 0:
        # 所有 complement 变量必须为假
        for v in complement:
            clauses.append([-v])
        return clauses, aux

    seq_clauses, aux = _sequential_counter_encode(complement, max_true, aux)
    clauses.extend(seq_clauses)
    return clauses, aux


def warners_encode(pb_constraints, num_vars):
    """
    Warners 编码 (加法器网络 / 顺序计数器)

    """
    clauses = []
    aux = num_vars + 1

    for coeffs, lits, op, bound in pb_constraints:
        if op != '>=':
            raise ValueError(f"Unsupported operator: {op}")

        z_vars = []
        z_coeffs = []
        for coeff, lit in zip(coeffs, lits):
            c = int(round(coeff))
            if c <= 0:
                continue
            z = aux
            aux += 1
            z_vars.append(z)
            z_coeffs.append(c)
            # 等价约束: z ↔ lit
            orig = abs(lit)
            if lit > 0:
                # z ↔ x_orig:  ¬z ∨ x,  z ∨ ¬x
                clauses.append([-z, orig])
                clauses.append([z, -orig])
            else:
                # z ↔ ¬x_orig:  ¬z ∨ ¬x,  z ∨ x
                clauses.append([-z, -orig])
                clauses.append([z, orig])

        bound_int = int(round(bound))
        if bound_int <= 0:
            continue  # 永真约束

        cons_clauses, aux = _pb_to_at_least_k(
            z_vars, z_coeffs, bound_int, aux)
        clauses.extend(cons_clauses)

    return aux - 1, clauses


# ============================================================
# BDD 编码
# ============================================================

class BDDNode:
    __slots__ = ('id', 'var', 'low', 'high')
    def __init__(self, id_val, var=0, low=None, high=None):
        self.id = id_val; self.var = var; self.low = low; self.high = high


def _build_bdd_for_constraint(vars_list, coeffs, bound, aux_start):
    """为 Σ coeffs[i]·x_i >= bound 构建 ROBDD (DP 自顶向下)"""
    n = len(vars_list); coeffs_int = [int(round(c)) for c in coeffs]
    bound_int = int(round(bound))
    TRUE = BDDNode(1, 0); FALSE = BDDNode(0, 0)
    memo, next_id = {}, [aux_start]

    def build(i, rem):
        if rem <= 0:
            return TRUE
        if i >= n:
            return FALSE
        key = (i, rem)
        if key in memo:
            return memo[key]
        lo = build(i + 1, rem)
        hi = build(i + 1, rem - coeffs_int[i])
        if lo is hi:
            node = lo
        else:
            node = BDDNode(next_id[0], vars_list[i], lo, hi)
            next_id[0] += 1
        memo[key] = node
        return node

    root = build(0, bound_int)
    return root, next_id[0]


def _bdd_to_cnf(root, aux_start):
    """ROBDD → CNF (ITE Tseitin 变换)"""
    clauses = []
    visited = set()

    def enc(node):
        if node.id <= 1 or node.id in visited:
            return
        visited.add(node.id)
        nid, v = node.id, node.var
        lo, hi = node.low.id, node.high.id

        # 6 Tseitin 子句 (已简化终端情况)
        if lo == 0:   clauses.append([-v, nid])
        elif lo != 1: clauses.append([-v, -lo, nid])
        if lo == 1:   clauses.append([-v, -nid])
        elif lo != 0: clauses.append([-v, lo, -nid])
        if hi == 0:   clauses.append([v, nid])
        elif hi != 1: clauses.append([v, -hi, nid])
        if hi == 1:   clauses.append([v, -nid])
        elif hi != 0: clauses.append([v, hi, -nid])
        enc(node.low); enc(node.high)

    enc(root)
    if root.id > 1:
        clauses.append([root.id])
    elif root.id == 0:
        clauses.append([aux_start, -aux_start])
    return clauses


def bdd_encode(pb_constraints, num_vars):
    """
    BDD 编码 (二元决策图 → Tseitin CNF)

    输入:
        pb_constraints: [(coeffs, lits, '>=', bound), ...]
        num_vars: 原始变量数
    输出:
        (new_num_vars, cnf_clauses)
    """
    clauses = []
    aux = num_vars + 1

    for coeffs, lits, op, bound in pb_constraints:
        if op != '>=':
            raise ValueError(f"Unsupported operator: {op}")

        # 为每个 literal 创建辅助变量 z_i ↔ lit_i
        z_vars = []
        z_coeffs = []
        for coeff, lit in zip(coeffs, lits):
            c = int(round(coeff))
            if c <= 0:
                continue
            z = aux
            aux += 1
            z_vars.append(z)
            z_coeffs.append(c)
            # 等价约束: z ↔ lit
            orig = abs(lit)
            if lit > 0:
                clauses.append([-z, orig])
                clauses.append([z, -orig])
            else:
                clauses.append([-z, -orig])
                clauses.append([z, orig])

        bound_int = int(round(bound))
        if bound_int <= 0:
            continue

        # 在 z_vars 上构建 BDD
        root, next_aux = _build_bdd_for_constraint(
            z_vars, z_coeffs, bound_int, aux)
        bdd_clauses = _bdd_to_cnf(root, aux)
        clauses.extend(bdd_clauses)
        aux = next_aux

    return aux - 1, clauses


def pb_to_cnf_file(pb_constraints, num_vars, output_path, method='warners'):
    """将 PB 公式写为 DIMACS CNF 文件"""
    if method == 'warners':
        nv, clauses = warners_encode(pb_constraints, num_vars)
    elif method == 'bdd':
        nv, clauses = bdd_encode(pb_constraints, num_vars)
    else:
        raise ValueError(f"Unknown method: {method}")

    with open(output_path, 'w') as f:
        f.write(f'p cnf {nv} {len(clauses)}\n')
        for clause in clauses:
            unique = sorted(set(clause), key=abs)
            f.write(' '.join(str(l) for l in unique) + ' 0\n')

    return nv, len(clauses)
