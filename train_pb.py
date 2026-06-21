"""
Attn-JGNN-PB 训练脚本 

数据集:
  - PB Competition 2006
  - Exact Cover
  - Wireless Sensor Network

用法:
    python train_pb.py                                    # 默认加载全部 PB 数据
    python train_pb.py --epochs 100 --lr 0.0005           # 自定义超参
"""
import os, sys, argparse, time, glob, random, pickle
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from config import Config
from models.attn_jgnn import AttnJGNN
from data.pb_parser import normalize_pb_constraints


def parse_simple_opb(filepath):
    """解析简化 OPB 格式"""
    num_vars = 0
    constraints = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('*'):
                if '#variable=' in line:
                    num_vars = int(line.split('#variable=')[1].split()[0])
                continue
            if not line or line.endswith(';') == False:
                continue
            # Skip objective function declarations (min: / max:)
            if line.startswith('min:') or line.startswith('max:'):
                continue
            # Remove "min:" or "max:" prefix if present on constraint lines
            line = line.rstrip(';')
            tokens = line.split()
            coeffs, lits = [], []
            i = 0
            while i < len(tokens):
                tok = tokens[i]
                if tok in ['>=', '<=', '=']:
                    op = tok
                    bound = int(tokens[i+1])
                    break
                # Parse term: +C xV or C ~xV
                if tok.startswith('+'):
                    coeff = int(tok[1:])
                    i += 1
                    lit_str = tokens[i]
                    if lit_str.startswith('~x'):
                        lit = -int(lit_str[2:])
                    else:
                        lit = int(lit_str[1:]) if lit_str.startswith('x') else int(lit_str)
                else:
                    # tok is a coefficient (no + sign before negative literal coeff)
                    coeff = int(tok)
                    i += 1
                    lit_str = tokens[i]
                    if lit_str.startswith('~x'):
                        lit = -int(lit_str[2:])
                    else:
                        lit = int(lit_str[1:]) if lit_str.startswith('x') else int(lit_str)
                coeffs.append(coeff)
                lits.append(lit)
                i += 1
            constraints.append((coeffs, lits, op, bound))
    return num_vars, constraints
from graph.factor_graph import build_factor_graph
from graph.join_graph import build_join_graph_adaptive


# ======================== 数据加载 ========================
def load_pb_data(data_base=None):
    """加载全部 PB 数据，保持原有 train/test 划分"""
    if data_base is None:
        data_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    train_items = []
    test_items = []

    for pname in ['PB_Competition2006', 'ExactCover', 'WSN']:
        for split in ['train', 'test']:
            dp = os.path.join(data_base, pname, split)
            if not os.path.exists(dp):
                continue
            for fn in os.listdir(dp):
                if not (fn.endswith('.opb') or fn.endswith('.wpbf')):
                    continue
                fp = os.path.join(dp, fn)
                try:
                    nv, raw = parse_simple_opb(fp)
                    cs = normalize_pb_constraints(raw, normalize_coeffs=True)
                    item = {'filepath': fp, 'filename': fn, 'num_vars': nv, 'constraints': cs}
                    if split == 'train':
                        train_items.append(item)
                    else:
                        test_items.append(item)
                except Exception as e:
                    print(f"  Skip {fn}: {e}")

    # 加载标签
    label_file = os.path.join(data_base, 'labels', 'pb_labels.pkl')
    labels = {}
    if os.path.exists(label_file):
        with open(label_file, 'rb') as f:
            labels = pickle.load(f)

    for item in train_items + test_items:
        item['true_count'] = labels.get(item['filename'], 0)

    print(f"[PB Data] Train: {len(train_items)} | Test: {len(test_items)}")
    return train_items, test_items


def build_graph(item, config):
    nv, cs = item['num_vars'], item['constraints']
    data = build_factor_graph(nv, cs, task='pb')
    data = build_join_graph_adaptive(data, config.flow_cutter_path, task='pb')
    tc = max(item['true_count'], 1)
    logZ = np.log(float(tc))
    # Cap extreme values to prevent gradient explosion
    MAX_LOGZ = 50.0
    logZ = min(logZ, MAX_LOGZ)
    data.true_logZ = torch.tensor(logZ, dtype=torch.float)
    return data


# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser(description='Attn-JGNN-PB Training')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = Config()
    config.task = 'pb'
    config.lr = args.lr
    config.epochs = args.epochs
    device = config.device

    print("=" * 60)
    print("Attn-JGNN-PB Training ")
    print(f"Device: {device} | Epochs: {args.epochs} | LR: {args.lr}")
    print("=" * 60)

    # 日志文件
    log_file = open("training_log.txt", "w", encoding="utf-8")

    # 加载数据
    train_items, test_items = load_pb_data()
    if len(train_items) == 0:
        print("No PB data found!"); sys.exit(1)

    # 初始化模型
    sample = build_graph(train_items[0], config)
    sample.true_logZ = torch.tensor(0.0)
    model = AttnJGNN(config, sample).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    best_rmse = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- 预构建所有图数据 (避免每轮重复树分解) ----
    print("Pre-building graphs...")
    train_graphs = [build_graph(item, config).to(device) for item in train_items]
    test_graphs  = [build_graph(item, config).to(device) for item in test_items]
    n_train = len(train_graphs)
    n_test  = len(test_graphs)
    print(f"  Train: {n_train} graphs, Test: {n_test} graphs")

    history = {'train_rmse': [], 'test_rmse': []}
    t0 = time.time()

    for epoch in range(args.epochs):
        # ======== Train ========
        model.train()
        train_rmse_sum = 0.0
        pbar = tqdm(train_graphs, desc=f"Epoch {epoch+1:3d}/{args.epochs} [Train]",
                    ncols=100, unit="inst", leave=False)
        for data in pbar:
            optimizer.zero_grad()
            pred_logZ, loss_dict = model(data, epoch=epoch)
            loss = loss_dict.get('total_loss')
            if loss is not None and loss.item() != 0.0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            rmse_val = np.sqrt(loss_dict.get('rmse_loss', 0.0))
            train_rmse_sum += rmse_val
            pbar.set_postfix(rmse=f"{rmse_val:.3f}")
        train_rmse = train_rmse_sum / n_train
        history['train_rmse'].append(train_rmse)
        scheduler.step()

        # ======== Eval ========
        model.eval()
        test_rmse_sum = 0.0
        with torch.no_grad():
            pbar2 = tqdm(test_graphs, desc=f"Epoch {epoch+1:3d}/{args.epochs} [Eval ]",
                         ncols=100, unit="inst", leave=False)
            for data in pbar2:
                pred_logZ, _ = model(data)
                diff = abs(pred_logZ.item() - data.true_logZ.item())
                test_rmse_sum += diff
                pbar2.set_postfix(diff=f"{diff:.3f}")
        test_rmse = test_rmse_sum / n_test
        history['test_rmse'].append(test_rmse)

        # ======== 输出 ========
        elapsed = time.time() - t0
        eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)

        improved = " *" if test_rmse < best_rmse else "  "
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            torch.save(model.state_dict(), os.path.join(args.save_dir, "attn_jgnn_pb_best.pt"))

        bar_filled  = "█" * int((epoch + 1) / args.epochs * 20)
        bar_empty   = "░" * (20 - len(bar_filled))

        print(f"  [{bar_filled}{bar_empty}] {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_rmse:.4f} | Test: {test_rmse:.4f} | "
              f"Best: {best_rmse:.4f}{improved} | "
              f"{elapsed:.0f}s (ETA: {eta:.0f}s)")

    # ======== 最终汇总 ========
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best  Test RMSE:  {best_rmse:.4f}")
    print(f"  Final Train RMSE: {history['train_rmse'][-1]:.4f}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
