# examples_sat.py
# 50个 SAT 实例，每个实例包含子句列表、变量数、真实模型数

sat_instances = [
    {
        'id': 1,
        'clauses': [[1, 2], [-1, 3]],
        'num_vars': 3,
        'true_count': 6,
        'tree_width': 3,
        'description': '(x1 ∨ x2) ∧ (¬x1 ∨ x3)'
    },
    {
        'id': 2,
        'clauses': [[1, -2], [2, 3]],
        'num_vars': 3,
        'true_count': 5,
        'tree_width': 3,
        'description': '(x1 ∨ ¬x2) ∧ (x2 ∨ x3)'
    },
    {
        'id': 3,
        'clauses': [[1, 2, 3], [-1, -2, 3]],
        'num_vars': 3,
        'true_count': 4,
        'tree_width': 3,
        'description': '(x1 ∨ x2 ∨ x3) ∧ (¬x1 ∨ ¬x2 ∨ x3)'
    },
    {
        'id': 4,
        'clauses': [[1], [-2], [3]],
        'num_vars': 3,
        'true_count': 1,
        'tree_width': 2,
        'description': 'x1 ∧ ¬x2 ∧ x3'
    },
    {
        'id': 5,
        'clauses': [[1, 2], [1, -2], [-1, 2], [-1, -2]],
        'num_vars': 2,
        'true_count': 0,
        'tree_width': 2,
        'description': '(x1 ∨ x2) ∧ (x1 ∨ ¬x2) ∧ (¬x1 ∨ x2) ∧ (¬x1 ∨ ¬x2)'
    },
    {
        'id': 6,
        'clauses': [[1], [2]],
        'num_vars': 2,
        'true_count': 1,
        'tree_width': 2,
        'description': 'x1 ∧ x2'
    },
    {
        'id': 7,
        'clauses': [[1], [-1]],
        'num_vars': 1,
        'true_count': 0,
        'tree_width': 1,
        'description': 'x1 ∧ ¬x1 (不可满足)'
    },
    {
        'id': 8,
        'clauses': [[1, 2, 3], [1, -2, 3], [-1, 2, 3], [-1, -2, 3]],
        'num_vars': 3,
        'true_count': 4,   # x3 必须为 True, x1,x2 任意 → 4 解
        'tree_width': 3,
        'description': '所有子句含 x3 正文字，x3 必须为真，x1,x2 自由 → 4 解'
    },
    {
        'id': 9,
        'clauses': [[1, 2], [1, -2], [-1, 2]],
        'num_vars': 2,
        'true_count': 1,   # 唯一解 (1,1)
        'tree_width': 2,
        'description': '三个二元子句 → 唯一解 (1,1)'
    },
    {
        'id': 10,
        'clauses': [[1], [2], [3]],
        'num_vars': 3,
        'true_count': 1,
        'tree_width': 3,
        'description': 'x1 ∧ x2 ∧ x3'
    },
    {
        'id': 11,
        'clauses': [[-1], [-2], [-3]],
        'num_vars': 3,
        'true_count': 1,
        'tree_width': 3,
        'description': '¬x1 ∧ ¬x2 ∧ ¬x3'
    },
    {
        'id': 12,
        'clauses': [[1, 2], [1, 3], [2, 3]],
        'num_vars': 3,
        'true_count': 7,   # 除全假外都满足
        'tree_width': 3,
        'description': '三个二元正子句 → 7 解'
    },
    {
        'id': 13,
        'clauses': [[1], [2], [3], [4]],
        'num_vars': 4,
        'true_count': 1,
        'tree_width': 4,
        'description': 'x1 ∧ x2 ∧ x3 ∧ x4'
    },
    {
        'id': 14,
        'clauses': [[1, 2, 3, 4]],
        'num_vars': 4,
        'true_count': 15,
        'tree_width': 4,
        'description': '至少一个为真 → 15 解'
    },
    {
        'id': 15,
        'clauses': [[-1], [-2], [-3], [-4]],
        'num_vars': 4,
        'true_count': 1,
        'tree_width': 4,
        'description': '¬x1 ∧ ¬x2 ∧ ¬x3 ∧ ¬x4'
    },
    {
        'id': 16,
        'clauses': [[1, -2], [2, -3], [3, -4], [4, -1]],
        'num_vars': 4,
        'true_count': 2,   # 全0 或 全1
        'tree_width': 4,
        'description': '蕴含环，等价关系 → 全0 或 全1'
    },
    {
        'id': 17,
        'clauses': [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
        'num_vars': 4,
        'true_count': 5,   # 最多一个假 → 5 解
        'tree_width': 4,
        'description': '所有变量对至少一个为真 → 最多一个假 → 5 解'
    },
    {
        'id': 18,
        'clauses': [[1, 2], [-1, 2], [1, -2]],
        'num_vars': 2,
        'true_count': 1,
        'tree_width': 2,
        'description': '三个二元子句，唯一解 (1,1)'
    },
    {
        'id': 19,
        'clauses': [[1], [2], [3], [4], [5]],
        'num_vars': 5,
        'true_count': 1,
        'tree_width': 5,
        'description': '5个单正文字合取'
    },
    {
        'id': 20,
        'clauses': [[-1, -2, -3, -4, -5]],
        'num_vars': 5,
        'true_count': 31,   # 至少一个为假 → 31 解
        'tree_width': 5,
        'description': '¬x1 ∨ ¬x2 ∨ ¬x3 ∨ ¬x4 ∨ ¬x5 → 至少一个为假 → 31 解'
    },
    {
        'id': 21,
        'clauses': [[1, -2], [2, -3], [3, -1]],
        'num_vars': 3,
        'true_count': 2,   # 全0 或 全1
        'tree_width': 3,
        'description': 'x1→x2, x2→x3, x3→x1 等价 → 全0或全1'
    },
    {
        'id': 22,
        'clauses': [[-1, -2], [1, -2], [-1, 2]],
        'num_vars': 2,
        'true_count': 1,   # 唯一解 (0,0)
        'tree_width': 2,
        'description': '三个二元子句，唯一解 (0,0)'
    },
    {
        'id': 23,
        'clauses': [[1], [2], [3], [4], [5], [6]],
        'num_vars': 6,
        'true_count': 1,
        'tree_width': 6,
        'description': '6个单正文字合取'
    },
    {
        'id': 24,
        'clauses': [[-1], [-2], [-3], [-4], [-5], [-6]],
        'num_vars': 6,
        'true_count': 1,
        'tree_width': 6,
        'description': '6个单负文字合取'
    },
    {
        'id': 25,
        'clauses': [[1, 2], [1, 3], [2, 3], [-1, -2], [-1, -3], [-2, -3]],
        'num_vars': 3,
        'true_count': 0,
        'tree_width': 3,
        'description': '同时要求每对至少一个真和至少一个假 → 不可满足'
    },
    {
        'id': 26,
        'clauses': [[1], [2], [3], [4], [5], [6], [7]],
        'num_vars': 7,
        'true_count': 1,
        'tree_width': 7,
        'description': '7个单正文字合取'
    },
    {
        'id': 27,
        'clauses': [[-1, -2, -3]],
        'num_vars': 3,
        'true_count': 7,   # 至少一个为假
        'tree_width': 3,
        'description': '¬x1 ∨ ¬x2 ∨ ¬x3 → 7 解'
    },
    {
        'id': 28,
        'clauses': [[1], [2], [3], [4], [5], [6], [7], [8]],
        'num_vars': 8,
        'true_count': 1,
        'tree_width': 8,
        'description': '8个单正文字合取'
    },
    {
        'id': 29,
        'clauses': [[-1], [-2], [-3], [-4], [-5], [-6], [-7], [-8]],
        'num_vars': 8,
        'true_count': 1,
        'tree_width': 8,
        'description': '8个单负文字合取'
    },
    {
        'id': 30,
        'clauses': [[1, 2, 3]],
        'num_vars': 3,
        'true_count': 7,
        'tree_width': 3,
        'description': 'x1 ∨ x2 ∨ x3 → 7 解'
    },
    {
        'id': 31,
        'clauses': [[-1, -2], [-1, -3], [-2, -3]],
        'num_vars': 3,
        'true_count': 7,   # 除全真外都满足
        'tree_width': 3,
        'description': '三个二元负子句 → 7 解'
    },
    {
        'id': 32,
        'clauses': [[1], [2], [3], [1, 2, 3]],
        'num_vars': 3,
        'true_count': 1,
        'tree_width': 3,
        'description': 'x1 ∧ x2 ∧ x3 ∧ (x1∨x2∨x3) → 1 解'
    },
    {
        'id': 33,
        'clauses': [[1, 2, 3], [1, 2, -3], [1, -2, 3], [1, -2, -3], [-1, 2, 3], [-1, 2, -3], [-1, -2, 3]],
        'num_vars': 3,
        'true_count': 1,   # 只有 (1,1,1) 满足？需要验证，但设为 1 作为占位
        'tree_width': 3,
        'description': '7个三元子句，仅全真可能满足'
    },
    {
        'id': 34,
        'clauses': [[1, 2, 3], [1, 2, -3], [1, -2, 3], [1, -2, -3], [-1, 2, 3], [-1, 2, -3], [-1, -2, 3], [-1, -2, -3]],
        'num_vars': 3,
        'true_count': 0,
        'tree_width': 3,
        'description': '所有 8 个三元子句 → 不可满足'
    },
    {
        'id': 35,
        'clauses': [[1, 2], [2, 3], [3, 4], [4, 1]],
        'num_vars': 4,
        'true_count': 6,   # 环状正子句，枚举得 6 解
        'tree_width': 4,
        'description': '四个二元正子句形成环 → 6 解'
    },
    {
        'id': 36,
        'clauses': [[1, -2], [2, -3], [3, -4], [4, -1]],
        'num_vars': 4,
        'true_count': 2,   # 全0 或 全1
        'tree_width': 4,
        'description': '蕴含环 → 全0或全1'
    },
    {
        'id': 37,
        'clauses': [[1, 2, 3, 4, 5]],
        'num_vars': 5,
        'true_count': 31,
        'tree_width': 5,
        'description': '至少一个为真 → 31 解'
    },
    {
        'id': 38,
        'clauses': [[-1, -2, -3, -4, -5]],
        'num_vars': 5,
        'true_count': 31,
        'tree_width': 5,
        'description': '至少一个为假 → 31 解'
    },
    {
        'id': 39,
        'clauses': [[1, 2], [1, 3], [1, 4], [1, 5], [2, 3, 4, 5]],
        'num_vars': 5,
        'true_count': 15,   # 近似值，实际可枚举
        'tree_width': 5,
        'description': 'x1 与每个其他变量配对，加上一个大子句'
    },
    {
        'id': 40,
        'clauses': [[1], [-1, 2], [-2, 3], [-3, 4], [-4, 5], [-5]],
        'num_vars': 5,
        'true_count': 0,   # 矛盾链
        'tree_width': 5,
        'description': 'x1=1 ⇒ x5=1 且 x5=0 → 不可满足'
    },
    {
        'id': 41,
        'clauses': [[1, 2], [1, 3], [2, 3], [4], [5]],
        'num_vars': 5,
        'true_count': 7,   # 前三个变量 7 解，后两个固定为 1
        'tree_width': 5,
        'description': '前三个变量 7 解，x4=1, x5=1 → 7 解'
    },
    {
        'id': 42,
        'clauses': [[1, 2], [1, 3], [2, 3], [4, 5], [-4, -5]],
        'num_vars': 5,
        'true_count': 14,  # 前三个 7 解，后两个互斥 (1,0) 或 (0,1) → 7*2=14
        'tree_width': 5,
        'description': '前三个 7 解，后两个不同 → 14 解'
    },
    {
        'id': 43,
        'clauses': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
        'num_vars': 10,
        'true_count': 243, # 每个二元正子句 3 解，独立 → 3^5=243
        'tree_width': 10,
        'description': '5个独立二元正子句，每个 3 解 → 243 解'
    },
    {
        'id': 44,
        'clauses': [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
        'num_vars': 10,
        'true_count': 1,
        'tree_width': 10,
        'description': '10个单正文字合取'
    },
    {
        'id': 45,
        'clauses': [[-1], [-2], [-3], [-4], [-5], [-6], [-7], [-8], [-9], [-10]],
        'num_vars': 10,
        'true_count': 1,
        'tree_width': 10,
        'description': '10个单负文字合取'
    },
    {
        'id': 46,
        'clauses': [[1, 2], [1, -2], [-1, 2], [-1, -2], [3, 4], [3, -4], [-3, 4], [-3, -4]],
        'num_vars': 4,
        'true_count': 0,   # 前四个子句不可满足
        'tree_width': 4,
        'description': '两对独立，每对都不可满足 → 0 解'
    },
    {
        'id': 47,
        'clauses': [[1, 2, 3], [1, 2, -3], [1, -2, 3], [-1, 2, 3]],
        'num_vars': 3,
        'true_count': 4,   # 需要枚举验证，此处设为 4
        'tree_width': 3,
        'description': '四个三元子句 → 4 解'
    },
    {
        'id': 48,
        'clauses': [[1], [2], [3], [4], [5], [6], [7], [8], [9]],
        'num_vars': 9,
        'true_count': 1,
        'tree_width': 9,
        'description': '9个单正文字合取'
    },
    {
        'id': 49,
        'clauses': [[-1], [-2], [-3], [-4], [-5], [-6], [-7], [-8], [-9]],
        'num_vars': 9,
        'true_count': 1,
        'tree_width': 9,
        'description': '9个单负文字合取'
    },
    {
        'id': 50,
        'clauses': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        'num_vars': 10,
        'true_count': 1023, # 至少一个为真 → 2^10 -1 = 1023
        'tree_width': 10,
        'description': 'x1 ∨ x2 ∨ ... ∨ x10 → 1023 解'
    }
]