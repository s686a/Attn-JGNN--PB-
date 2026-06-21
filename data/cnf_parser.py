def parse_cnf(filepath):
    """
    解析 DIMACS CNF 文件，返回 (num_vars, clauses)
    clauses: list of list of int, 每个子句由文字组成，正数表示正文字，负数表示负文字
    """
    num_vars = 0
    clauses = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('c'):
                continue
            if line.startswith('p'):
                parts = line.split()
                num_vars = int(parts[2])
            else:
                if line.endswith(' 0'):
                    line = line[:-2]
                lits = list(map(int, line.split()))
                if lits:
                    clauses.append(lits)
    return num_vars, clauses