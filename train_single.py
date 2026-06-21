
import argparse
import torch
import torch.optim as optim
import numpy as np
import random
import os
import sys

# 导入自定义模块
from config import Config
from data.cnf_parser import parse_cnf
from data.pb_parser import parse_opb
from graph.factor_graph import build_factor_graph
from graph.join_graph import build_join_graph_adaptive   # 自适应树分解版本
from models.attn_jgnn import AttnJGNN
from utils.metrics import compute_all_metrics

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='Attn-JGNN/Attn-PB Training')
    parser.add_argument('--task', type=str, choices=['sat', 'pb'], required=True,
                        help='Task type: sat (CNF) or pb (Pseudo-Boolean)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file path (.cnf for sat, .opb for pb)')
    parser.add_argument('--true_logz', type=float, default=None,
                        help='Ground truth log model count (for supervised training)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--flow_cutter_path', type=str, default='./flow_cutter_pace17',
                        help='Path to FlowCutter executable (for tree decomposition)')
    args = parser.parse_args()

    set_seed(args.seed)

    # 加载配置
    config = Config()
    config.task = args.task
    config.lr = args.lr
    config.epochs = args.epochs
    config.flow_cutter_path = args.flow_cutter_path

    # 1. 解析输入文件
    if args.task == 'sat':
        num_vars, clauses = parse_cnf(args.input)
        data = build_factor_graph(num_vars, clauses, task='sat')
    else:
        num_vars, constraints = parse_opb(args.input)
        # 注意：PB 约束需要预处理（统一为≥，系数归一化等）
        data = build_factor_graph(num_vars, constraints, task='pb')

    # 2. 自适应树分解（基于公式复杂度动态选择树宽）
    data = build_join_graph_adaptive(data, config.flow_cutter_path, task=args.task)
    print(f"Adaptive tree decomposition completed. Used treewidth: {data.used_treewidth}")

    # 3. 真值标签
    if args.true_logz is not None:
        data.true_logZ = torch.tensor(args.true_logz, dtype=torch.float)
    else:
        data.true_logZ = torch.tensor(0.0)
        print("Warning: true_logz not provided, using dummy value 0.0. Please set --true_logz for meaningful training.")

    # 4. 模型初始化
    model = AttnJGNN(config, data).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # 将数据移到设备
    data = data.to(config.device)

    # 5. 训练循环
    model.train()
    print(f"Starting training for {config.epochs} epochs...")
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        pred_logZ, loss_dict = model(data, epoch=epoch)
        loss = loss_dict.get('total_loss')
        if loss is not None and loss.item() != 0.0:
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: pred_logZ = {pred_logZ.item():.4f}, "
                  f"RMSE = {loss_dict.get('rmse_loss', 0.0):.4f}, "
                  f"Constraint = {loss_dict.get('constraint_loss', 0.0):.4f}")

    # 6. 最终评估
    model.eval()
    with torch.no_grad():
        final_pred, _ = model(data, epoch=config.epochs)
        final_rmse = (final_pred - data.true_logZ).pow(2).mean().sqrt().item()
        print(f"Training finished. Final pred_logZ = {final_pred.item():.4f}, "
              f"True logZ = {data.true_logZ.item():.4f}, RMSE = {final_rmse:.6f}")

if __name__ == '__main__':
    main()