
def normalize_pb_constraints(constraints, normalize_coeffs=True):
    """
    输入: list of (coeffs, lits, op, bound)
    输出: list of (coeffs, lits, '>=', bound)  所有约束统一为 >=，系数为正
    """
    processed = []
    # 先拆分等式
    for coeffs, lits, op, bound in constraints:
        if op == '=':
            processed.append((coeffs, lits, '>=', bound))
            processed.append((coeffs, lits, '<=', bound))
        else:
            processed.append((coeffs, lits, op, bound))

    normalized = []
    for coeffs, lits, op, bound in processed:
        # 负系数转正
        new_coeffs = []
        new_lits = []
        const_shift = 0
        for a, lit in zip(coeffs, lits):
            if a < 0:
                a_pos = -a
                lit = -lit
                const_shift += a_pos
                new_coeffs.append(a_pos)
            else:
                new_coeffs.append(a)
            new_lits.append(lit)
        new_bound = bound + const_shift

        if op == '<=':
            # 转化为 >=
            new_coeffs = [-c for c in new_coeffs]
            new_bound = -new_bound
            # 转换后可能又有负系数，递归处理
            sub = normalize_pb_constraints([(new_coeffs, new_lits, '>=', new_bound)], normalize_coeffs=False)
            normalized.extend(sub)
        else:  # '>='
            normalized.append((new_coeffs, new_lits, '>=', new_bound))

    # 确保系数为正
    final = []
    for coeffs, lits, op, bound in normalized:
        if any(c < 0 for c in coeffs):
            # 再次负系数转正
            new_coeffs = []
            new_lits = []
            const_shift = 0
            for a, lit in zip(coeffs, lits):
                if a < 0:
                    a_pos = -a
                    lit = -lit
                    const_shift += a_pos
                    new_coeffs.append(a_pos)
                else:
                    new_coeffs.append(a)
                new_lits.append(lit)
            bound = bound + const_shift
            final.append((new_coeffs, new_lits, '>=', bound))
        else:
            final.append((coeffs, lits, '>=', bound))

    # 归一化
    if normalize_coeffs:
        all_coeffs = [c for cons in final for c in cons[0]]
        if all_coeffs:
            max_c = max(all_coeffs)
            if max_c > 0:
                normed = []
                for coeffs, lits, op, bound in final:
                    normed_coeffs = [c / max_c for c in coeffs]
                    normed_bound = bound / max_c
                    normed.append((normed_coeffs, lits, op, normed_bound))
                return normed
    return final

def parse_opb(filepath, normalize=True):
    """
    解析 OPB 文件，返回 (num_vars, constraints)
    constraints: list of (coeffs, lits, bound) 已经标准化为 >= 形式且系数为正
    """
    num_vars = 0
    raw_constraints = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('*'):
                # 查找 #variable= 信息
                if '#variable=' in line:
                    num_vars = int(line.split('#variable=')[1].split()[0])
                continue
            if not line:
                continue
            if line.endswith(';'):
                line = line[:-1]
            tokens = line.split()
            coeffs = []
            lits = []
            op = None
            bound = None
            i = 0
            while i < len(tokens):
                tok = tokens[i]
                if tok in ['>=', '<=', '=']:
                    op = tok
                    bound = int(tokens[i+1])
                    break
                coeff = int(tok)
                lit_str = tokens[i+1]
                if lit_str.startswith('x'):
                    lit = int(lit_str[1:])
                elif lit_str.startswith('~x'):
                    lit = -int(lit_str[2:])
                else:
                    lit = int(lit_str)
                coeffs.append(coeff)
                lits.append(lit)
                i += 2
            raw_constraints.append((coeffs, lits, op, bound))

    # 标准化预处理
    if normalize:
        constraints = normalize_pb_constraints(raw_constraints, normalize_coeffs=True)
    else:
        # 如果不标准化，也需转换为 (coeffs, lits, bound) 格式（去掉 op，因为 op 已隐含为 >=）
        constraints = []
        for coeffs, lits, op, bound in raw_constraints:
            if op == '<=':
                # 转换为 >=
                coeffs = [-c for c in coeffs]
                bound = -bound
            elif op == '=':
                # 拆成两个约束
                constraints.append((coeffs[:], lits[:], bound))
                constraints.append(([-c for c in coeffs], lits[:], -bound))
                continue
            constraints.append((coeffs, lits, bound))

    return num_vars, constraints