"""
Attn-JGNN 评估脚本
支持 SAT 和 PB 任务的批量评估，输出 RMSE 及论文所需的各基线对比指标。

用法:
    # SAT 评估
    python evaluate.py --task sat --data_dir ./data/SATLIB --label_file ./data/satlib_labels.pkl --checkpoint ./checkpoints/attn_jgnn_satlib_best.pt

    # PB 评估
    python evaluate.py --task pb --data_dir ./data/ExactCover --label_file ./data/ec_labels.pkl --checkpoint ./checkpoints/attn_jgnn_pb_best.pt
"""
import os, sys, argparse, time, glob, pickle
import numpy as np
import torch
from config import Config
from models.attn_jgnn import AttnJGNN
from data.cnf_parser import parse_cnf
from data.pb_parser import parse_opb, normalize_pb_constraints
from graph.factor_graph import build_factor_graph
from graph.join_graph import build_join_graph_adaptive
from utils.metrics import compute_all_metrics


def load_sat_data(data_dir, label_file):
    """加载 SAT 数据"""
    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.cnf'), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, '*.cnf')))
    labels = {}
    if label_file and os.path.exists(label_file):
        with open(label_file, 'rb') as f:
            labels = pickle.load(f)
    instances = []
    for fp in files:
        fn = os.path.basename(fp)
        nv, cls = parse_cnf(fp)
        tc = labels.get(fn, labels.get(fp, 0))
        instances.append({'filepath': fp, 'filename': fn, 'num_vars': nv, 'clauses': cls, 'true_count': tc, 'type': 'sat'})
    return instances


def load_pb_data(data_dir, label_file):
    """加载 PB 数据"""
    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.opb'), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, '**', '*.wpbf'), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, '**', '*.pb'), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, '*.opb')))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, '*.wpbf')))
    labels = {}
    if label_file and os.path.exists(label_file):
        with open(label_file, 'rb') as f:
            labels = pickle.load(f)
    instances = []
    for fp in files:
        fn = os.path.basename(fp)
        nv, raw = parse_opb(fp, normalize=False)
        cs = normalize_pb_constraints(raw, normalize_coeffs=True)
        tc = labels.get(fn, labels.get(fp, 0))
        instances.append({'filepath': fp, 'filename': fn, 'num_vars': nv, 'constraints': cs, 'true_count': tc, 'type': 'pb'})
    return instances


def build_data(inst, config):
    """构建图数据"""
    if inst['type'] == 'sat':
        data = build_factor_graph(inst['num_vars'], inst['clauses'], task='sat')
    else:
        data = build_factor_graph(inst['num_vars'], inst['constraints'], task='pb')
    data = build_join_graph_adaptive(data, config.flow_cutter_path, task=inst['type'])
    tc = max(inst['true_count'], 1)
    data.true_logZ = torch.tensor(np.log(float(tc)), dtype=torch.float)
    return data


def main():
    parser = argparse.ArgumentParser(description='Attn-JGNN Evaluation')
    parser.add_argument('--task', type=str, required=True, choices=['sat', 'pb'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--label_file', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点 .pt')
    parser.add_argument('--output', type=str, default='./results.txt')
    args = parser.parse_args()

    config = Config()
    config.task = args.task

    print("=" * 60)
    print(f"Attn-JGNN Evaluation | Task: {args.task}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 60)

    # 数据
    if args.task == 'sat':
        instances = load_sat_data(args.data_dir, args.label_file)
    else:
        instances = load_pb_data(args.data_dir, args.label_file)
    print(f"Loaded {len(instances)} instances")

    if len(instances) == 0:
        print("No instances found!"); return

    # 模型
    sample = build_data(instances[0], config)
    sample.true_logZ = torch.tensor(0.0)
    model = AttnJGNN(config, sample).to(config.device)
    ckpt = torch.load(args.checkpoint, map_location=config.device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # 评估
    pred_list, true_list = [], []
    t0 = time.time()
    for idx, inst in enumerate(instances):
        data = build_data(inst, config)
        data = data.to(config.device)
        with torch.no_grad():
            pred_logZ, _ = model(data)
        pred_list.append(pred_logZ.item())
        true_list.append(data.true_logZ.item())

        if (idx + 1) % 50 == 0:
            print(f"  progress: {idx+1}/{len(instances)}")

    elapsed = time.time() - t0

    # 计算指标
    metrics = compute_all_metrics(pred_list, true_list)
    avg_time = elapsed / len(instances)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Instances evaluated: {len(instances)}")
    print(f"  RMSE:               {metrics['RMSE']:.4f}")
    print(f"  Within ±0.5:        {metrics['within_0.5']:.2%}")
    print(f"  Within ±1.0:        {metrics['within_1.0']:.2%}")
    print(f"  Mean Relative Error: {metrics['mean_relative_error']:.4f}")
    print(f"  Total time:          {elapsed:.1f}s")
    print(f"  Avg time/instance:   {avg_time:.3f}s")

    # 保存结果
    with open(args.output, 'w') as f:
        f.write(f"Attn-JGNN Evaluation | Task: {args.task}\n")
        f.write(f"Instances: {len(instances)}\n")
        f.write(f"RMSE: {metrics['RMSE']:.4f}\n")
        f.write(f"Within_0.5: {metrics['within_0.5']:.4f}\n")
        f.write(f"Within_1.0: {metrics['within_1.0']:.4f}\n")
        f.write(f"Mean_Relative_Error: {metrics['mean_relative_error']:.4f}\n")
        f.write(f"Total_Time_s: {elapsed:.1f}\n")
        f.write(f"Avg_Time_per_Instance_s: {avg_time:.3f}\n")
        f.write("\nPer-instance:\n")
        for i, (p, t) in enumerate(zip(pred_list, true_list)):
            f.write(f"{instances[i]['filename']}: pred={p:.4f}, true={t:.4f}\n")
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
