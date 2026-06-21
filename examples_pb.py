# examples_pb.py
# 50个伪布尔 (PB) 实例，每个实例包含约束列表、变量数、真实模型数

pb_instances = [
    {
        'id': 1,
        'constraints': [
            ([2, 1], [1, 2], '>=', 2),
            ([1, 1], [1, 3], '<=', 1)
        ],
        'num_vars': 3,
        'true_count': 2,
        'tree_width': 3,
        'description': '(2x1 + x2 >= 2) ∧ (x1 + x3 <= 1) → (1,0,0),(1,1,0)'
    },
    {
        'id': 2,
        'constraints': [
            ([1, 1], [1, 2], '>=', 1),
            ([1, 1], [1, 3], '<=', 1)
        ],
        'num_vars': 3,
        'true_count': 4,
        'tree_width': 3,
        'description': '(x1 + x2 >= 1) ∧ (x1 + x3 <= 1)'
    },
    {
        'id': 3,
        'constraints': [([2, 2], [1, 2], '>=', 3)],
        'num_vars': 2,
        'true_count': 1,
        'tree_width': 2,
        'description': '2x1 + 2x2 >= 3 → 唯一解 (1,1)'
    },
    {
        'id': 4,
        'constraints': [
            ([3, 2], [1, 2], '>=', 4),
            ([1, 1], [1, 2], '<=', 1)
        ],
        'num_vars': 2,
        'true_count': 0,
        'tree_width': 2,
        'description': '3x1+2x2>=4 且 x1+x2<=1 → 矛盾，无解'
    },
    {
        'id': 5,
        'constraints': [
            ([1, 1], [1, 2], '>=', 1),
            ([1, 1], [1, 2], '<=', 1)
        ],
        'num_vars': 2,
        'true_count': 2,
        'tree_width': 2,
        'description': 'x1 + x2 = 1 → (1,0), (0,1)'
    },
    {
        'id': 6,
        'constraints': [([1], [1], '>=', 1)],
        'num_vars': 1,
        'true_count': 1,
        'tree_width': 1,
        'description': 'x1 = 1'
    },
    {
        'id': 7,
        'constraints': [([1], [1], '<=', 0)],
        'num_vars': 1,
        'true_count': 1,
        'tree_width': 1,
        'description': 'x1 = 0'
    },
    {
        'id': 8,
        'constraints': [([1], [1], '>=', 1), ([1], [1], '<=', 0)],
        'num_vars': 1,
        'true_count': 0,
        'tree_width': 1,
        'description': 'x1=1 且 x1=0 → 矛盾'
    },
    {
        'id': 9,
        'constraints': [([1, 1], [1, 2], '>=', 1)],
        'num_vars': 2,
        'true_count': 3,
        'tree_width': 2,
        'description': 'x1 + x2 >= 1 → 除 (0,0) 外都满足 → 3解'
    },
    {
        'id': 10,
        'constraints': [([1, 1], [1, 2], '<=', 1)],
        'num_vars': 2,
        'true_count': 3,
        'tree_width': 2,
        'description': 'x1 + x2 <= 1 → 除 (1,1) 外都满足 → 3解'
    },
    {
        'id': 11,
        'constraints': [([1, 1], [1, 2], '=', 1)],
        'num_vars': 2,
        'true_count': 2,
        'tree_width': 2,
        'description': 'x1 + x2 = 1 → (1,0), (0,1)'
    },
    {
        'id': 12,
        'constraints': [([2, 1], [1, 2], '>=', 2)],
        'num_vars': 2,
        'true_count': 2,
        'tree_width': 2,
        'description': '2x1 + x2 >= 2 → (1,0), (1,1)'
    },
    {
        'id': 13,
        'constraints': [([1, 2], [1, 2], '>=', 2)],
        'num_vars': 2,
        'true_count': 2,
        'tree_width': 2,
        'description': 'x1 + 2x2 >= 2 → (0,1), (1,1)'
    },
    {
        'id': 14,
        'constraints': [([1, 1, 1], [1, 2, 3], '>=', 2)],
        'num_vars': 3,
        'true_count': 4,
        'tree_width': 3,
        'description': 'x1+x2+x3 >= 2 → 至少两个为真 → C(3,2)+C(3,3)=4解'
    },
    {
        'id': 15,
        'constraints': [([1, 1, 1], [1, 2, 3], '<=', 1)],
        'num_vars': 3,
        'true_count': 4,
        'tree_width': 3,
        'description': 'x1+x2+x3 <= 1 → 至多一个为真 → 4解'
    },
    {
        'id': 16,
        'constraints': [([1, 1, 1], [1, 2, 3], '=', 2)],
        'num_vars': 3,
        'true_count': 3,
        'tree_width': 3,
        'description': 'x1+x2+x3 = 2 → 恰好两个为真 → 3解'
    },
    {
        'id': 17,
        'constraints': [([2, 2, 2], [1, 2, 3], '>=', 3)],
        'num_vars': 3,
        'true_count': 4,
        'tree_width': 3,
        'description': '2x1+2x2+2x3 >= 3 → 至少两个为真 → 4解'
    },
    {
        'id': 18,
        'constraints': [([1, 1], [1, 2], '>=', 1), ([1, 1], [1, 2], '<=', 1)],
        'num_vars': 2,
        'true_count': 2,
        'tree_width': 2,
        'description': 'x1+x2 = 1 → 2解'
    },
    {
        'id': 19,
        'constraints': [([1, 1], [1, 2], '>=', 2)],
        'num_vars': 2,
        'true_count': 1,
        'tree_width': 2,
        'description': 'x1+x2 >= 2 → 唯一解 (1,1)'
    },
    {
        'id': 20,
        'constraints': [([1, 1], [1, 2], '<=', 0)],
        'num_vars': 2,
        'true_count': 1,
        'tree_width': 2,
        'description': 'x1+x2 <= 0 → 唯一解 (0,0)'
    },
    {
        'id': 21,
        'constraints': [([1, 1, 1], [1, 2, 3], '>=', 1), ([1, 1, 1], [1, 2, 3], '<=', 2)],
        'num_vars': 3,
        'true_count': 6,
        'tree_width': 3,
        'description': '1 ≤ x1+x2+x3 ≤ 2 → 6解'
    },
    {
        'id': 22,
        'constraints': [([3, 1], [1, 2], '>=', 2), ([1, 1], [1, 2], '<=', 1)],
        'num_vars': 2,
        'true_count': 1,
        'tree_width': 2,
        'description': '3x1+x2>=2 且 x1+x2<=1 → 解 (1,0)'
    },
    {
        'id': 23,
        'constraints': [([2, 3], [1, 2], '>=', 4), ([1, 1], [1, 2], '>=', 1)],
        'num_vars': 2,
        'true_count': 1,
        'tree_width': 2,
        'description': '2x1+3x2>=4 且 x1+x2>=1 → 解 (1,1)'
    },
    {
        'id': 24,
        'constraints': [([1, 1, 1], [1, 2, 3], '>=', 2), ([1, 1, 1], [1, 2, 3], '<=', 2)],
        'num_vars': 3,
        'true_count': 3,
        'tree_width': 3,
        'description': 'x1+x2+x3 = 2 → 3解'
    },
    {
        'id': 25,
        'constraints': [([1, 1], [1, 2], '>=', 1), ([1, 1], [2, 3], '>=', 1)],
        'num_vars': 3,
        'true_count': 5,
        'tree_width': 3,
        'description': 'x1+x2>=1 且 x2+x3>=1 → 5解'
    },
    {
        'id': 26,
        'constraints': [([1, 1], [1, 2], '>=', 1), ([1], [3], '=', 0)],
        'num_vars': 3,
        'true_count': 3,
        'tree_width': 3,
        'description': 'x1+x2>=1 且 x3=0 → (1,0,0),(0,1,0),(1,1,0) → 3解'
    },
    {
        'id': 27,
        'constraints': [([1, 1], [1, 2], '<=', 1), ([1, 1], [2, 3], '<=', 1)],
        'num_vars': 3,
        'true_count': 5,
        'tree_width': 3,
        'description': 'x1+x2<=1 且 x2+x3<=1 → 5解'
    },
    {
        'id': 28,
        'constraints': [([1], [1], '>=', 1), ([1], [2], '>=', 1), ([1], [3], '>=', 1)],
        'num_vars': 3,
        'true_count': 1,
        'tree_width': 3,
        'description': 'x1=1, x2=1, x3=1 → 1解'
    },
    {
        'id': 29,
        'constraints': [([1], [1], '<=', 0), ([1], [2], '<=', 0), ([1], [3], '<=', 0)],
        'num_vars': 3,
        'true_count': 1,
        'tree_width': 3,
        'description': 'x1=0, x2=0, x3=0 → 1解'
    },
    {
        'id': 30,
        'constraints': [([1, 1, 1, 1], [1, 2, 3, 4], '>=', 3)],
        'num_vars': 4,
        'true_count': 5,
        'tree_width': 4,
        'description': 'x1+x2+x3+x4 >= 3 → 至少3个真 → 5解'
    },
    {
        'id': 31,
        'constraints': [([1, 1, 1, 1], [1, 2, 3, 4], '<=', 1)],
        'num_vars': 4,
        'true_count': 5,
        'tree_width': 4,
        'description': '至多1个真 → 5解'
    },
    {
        'id': 32,
        'constraints': [([2, 2], [1, 2], '>=', 2), ([1, 1], [1, 2], '<=', 1)],
        'num_vars': 2,
        'true_count': 2,
        'tree_width': 2,
        'description': '等价于 x1+x2=1 → (1,0),(0,1) → 2解'
    },
    {
        'id': 33,
        'constraints': [([1, 2, 3], [1, 2, 3], '>=', 4)],
        'num_vars': 3,
        'true_count': 3,
        'tree_width': 3,
        'description': 'x1+2x2+3x3 >= 4 → (0,1,1),(1,0,1),(1,1,1) → 3解'
    },
    {
        'id': 34,
        'constraints': [([1, 1], [1, 2], '>=', 1), ([2, 1], [1, 2], '>=', 2)],
        'num_vars': 2,
        'true_count': 2,
        'tree_width': 2,
        'description': 'x1+x2>=1 且 2x1+x2>=2 → (1,0),(1,1) → 2解'
    },
    {
        'id': 35,
        'constraints': [([1, 1, 1], [1, 2, 3], '>=', 1), ([1, 1, 1], [1, 2, 3], '<=', 1)],
        'num_vars': 3,
        'true_count': 3,
        'tree_width': 3,
        'description': 'x1+x2+x3 = 1 → 3解'
    },
    {
        'id': 36,
        'constraints': [([1, 1, 1, 1, 1], [1, 2, 3, 4, 5], '>=', 4)],
        'num_vars': 5,
        'true_count': 6,
        'tree_width': 5,
        'description': '至少4个真 → C(5,4)+C(5,5)=6解'
    },
    {
        'id': 37,
        'constraints': [([1, 1, 1, 1, 1], [1, 2, 3, 4, 5], '<=', 1)],
        'num_vars': 5,
        'true_count': 6,
        'tree_width': 5,
        'description': '至多1个真 → 6解'
    },
    {
        'id': 38,
        'constraints': [([1, 1], [1, 2], '=', 0)],
        'num_vars': 2,
        'true_count': 1,
        'tree_width': 2,
        'description': 'x1=x2=0 → 1解'
    },
    {
        'id': 39,
        'constraints': [([1, 1], [1, 2], '=', 2)],
        'num_vars': 2,
        'true_count': 1,
        'tree_width': 2,
        'description': 'x1=x2=1 → 1解'
    },
    {
        'id': 40,
        'constraints': [([1, 1, 1], [1, 2, 3], '>=', 2), ([1, 1, 1], [1, 2, 3], '<=', 2)],
        'num_vars': 3,
        'true_count': 3,
        'tree_width': 3,
        'description': 'x1+x2+x3 = 2 → 3解'
    },
    {
        'id': 41,
        'constraints': [([1, 1], [1, 2], '>=', 1), ([1, 1], [2, 3], '>=', 1), ([1, 1], [3, 1], '>=', 1)],
        'num_vars': 3,
        'true_count': 1,
        'tree_width': 3,
        'description': '每对变量至少一个真 → 唯一解 (1,1,1)'
    },
    {
        'id': 42,
        'constraints': [([1, 1], [1, 2], '<=', 1), ([1, 1], [2, 3], '<=', 1), ([1, 1], [3, 1], '<=', 1)],
        'num_vars': 3,
        'true_count': 1,
        'tree_width': 3,
        'description': '每对变量至多一个真 → 唯一解 (0,0,0)'
    },
    {
        'id': 43,
        'constraints': [([1, 1, 1], [1, 2, 3], '>=', 1), ([1, 1, 1], [1, 2, 3], '>=', 2)],
        'num_vars': 3,
        'true_count': 4,
        'tree_width': 3,
        'description': 'x1+x2+x3 >= 2 → 4解'
    },
    {
        'id': 44,
        'constraints': [([1, 1, 1, 1], [1, 2, 3, 4], '=', 2)],
        'num_vars': 4,
        'true_count': 6,
        'tree_width': 4,
        'description': '恰好两个为真 → C(4,2)=6解'
    },
    {
        'id': 45,
        'constraints': [([2, 2, 2, 2], [1, 2, 3, 4], '>=', 6)],
        'num_vars': 4,
        'true_count': 5,
        'tree_width': 4,
        'description': '2(x1+x2+x3+x4) >= 6 → x1+...+x4 >= 3 → 5解'
    },
    {
        'id': 46,
        'constraints': [([3, 1, 1], [1, 2, 3], '>=', 4), ([1, 1], [1, 2], '<=', 1)],
        'num_vars': 3,
        'true_count': 1,
        'tree_width': 3,
        'description': '3x1+x2+x3>=4 且 x1+x2<=1 → 解 (1,0,1)'
    },
    {
        'id': 47,
        'constraints': [([1, 1], [1, 2], '>=', 1), ([1, 1], [2, 3], '>=', 1), ([1], [3], '=', 0)],
        'num_vars': 3,
        'true_count': 2,
        'tree_width': 3,
        'description': 'x1+x2>=1, x2+x3>=1, x3=0 → (0,1,0),(1,1,0)'
    },
    {
        'id': 48,
        'constraints': [([1, 1], [1, 2], '<=', 1), ([1, 1], [2, 3], '<=', 1), ([1], [3], '=', 1)],
        'num_vars': 3,
        'true_count': 2,
        'tree_width': 3,
        'description': 'x1+x2<=1, x2+x3<=1, x3=1 → (0,0,1),(1,0,1)'
    },
    {
        'id': 49,
        'constraints': [([1, 1, 1], [1, 2, 3], '>=', 2), ([1, 1], [2, 3], '<=', 1)],
        'num_vars': 3,
        'true_count': 2,
        'tree_width': 3,
        'description': 'x1+x2+x3>=2 且 x2+x3<=1 → (1,0,1),(1,1,0)'
    },
    {
        'id': 50,
        'constraints': [([1, 1], [1, 2], '=', 1), ([2, 2], [1, 2], '=', 2)],
        'num_vars': 2,
        'true_count': 2,
        'tree_width': 2,
        'description': 'x1+x2=1 且 2x1+2x2=2 → 等价条件 → (1,0),(0,1)'
    }
]