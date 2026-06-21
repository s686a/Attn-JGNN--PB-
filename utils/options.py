"""
命令行参数工具 (已由 config.py 统一管理)

注意: 此文件为早期实现, 参数定义已迁移到 config.py。
config.py 中对应参数:
  - dim = 64
  - max_iter = 5  (原 n_rounds = 10)
  - num_mlp_layers = 1  (原 n_mlp_layers = 3)

保留此文件作为命令行参数参考。
"""
import argparse


def add_model_options(parser):
    parser.add_argument('--model', type=str, default='AEIN', help='Model choice')
    parser.add_argument('--dim', type=int, default=64, help='Dimension of variable and clause embeddings')
    parser.add_argument('--n_rounds', type=int, default=10, help='Number of rounds of message passing (已废弃, 见 config.max_iter)')
    parser.add_argument('--n_mlp_layers', type=int, default=3, help='Number of layers in all MLPs (已废弃, 见 config.num_mlp_layers)')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function in all MLPs')
