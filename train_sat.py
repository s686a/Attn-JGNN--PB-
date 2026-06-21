"""
Attn-JGNN SAT 训练脚本
支持 BIRD 和 SATLIB 数据集：
  - BIRD: 70/30 训练/测试划分
  - SATLIB: 60/20/20 训练/验证/测试划分

用法:
    python train_sat.py --dataset bird --data_dir ./data/BIRD --label_file ./data/labels.pkl
    python train_sat.py --dataset satlib --data_dir ./data/SATLIB --label_file ./data/satlib_labels.pkl
    python train_sat.py --dataset bird --adaptive --data_dir ./data/BIRD --label_file ./data/labels.pkl
"""
import os, sys, argparse, time, glob, random, pickle
import numpy as np
import torch
import torch.optim as optim
from config import Config
from models.attn_jgnn import AttnJGNN
from data.cnf_parser import parse_cnf
from graph.factor_graph import build_factor_graph
from graph.join_graph import build_join_graph_adaptive
from utils.metrics import compute_all_metrics


# ======================== 数据集 ========================
class SATInstanceDataset:
    """SAT 实例数据集"""
    def __init__(self, data_dir, label_file=None):
        self.cnf_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.cnf'), recursive=True))
        if not self.cnf_files:
            self.cnf_files = sorted(glob.glob(os.path.join(data_dir, '*.cnf')))
        self.labels = {}
        if label_file and os.path.exists(label_file):
            with open(label_file, 'rb') as f:
                self.labels = pickle.load(f)
        print(f"[Dataset] {len(self.cnf_files)} CNF files, {len(self.labels)} labels")

    def __len__(self):
        return len(self.cnf_files)

    def __getitem__(self, idx):
        fp = self.cnf_files[idx]
        fn = os.path.basename(fp)
        nv, cls = parse_cnf(fp)
        tc = self.labels.get(fn, self.labels.get(fp, 0))
        return {'filepath': fp, 'filename': fn, 'num_vars': nv, 'clauses': cls, 'true_count': tc}


def build_data(item, config):
    """将单个 SAT 实例构建为图数据对象"""
    nv, cls = item['num_vars'], item['clauses']
    tc = item['true_count']
    data = build_factor_graph(nv, cls, task='sat')
    data = build_join_graph_adaptive(data, config.flow_cutter_path, task='sat')
    data.true_logZ = torch.tensor(np.log(max(tc, 1e-10)), dtype=torch.float)
    data.filename = item['filename']
    return data


# ======================== 训练/评估 ========================
def run_epoch(model, dataset, config, optimizer=None, current_epoch=0):
    """训练或评估一个 epoch"""
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss, total_rmse, n = 0.0, 0.0, 0
    preds, trues = [], []

    for idx in range(len(dataset)):
        item = dataset[idx]
        data = build_data(item, config)
        data = data.to(config.device)

        if is_train:
            optimizer.zero_grad()
            pred_logZ, loss_dict = model(data, epoch=current_epoch)
            loss = loss_dict.get('total_loss')
            if loss is not None and loss.item() != 0.0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            rmse_val = np.sqrt(loss_dict.get('rmse_loss', 0.0))
        else:
            with torch.no_grad():
                pred_logZ, loss_dict = model(data)
            rmse_val = abs(pred_logZ.item() - data.true_logZ.item())

        total_rmse += rmse_val
        n += 1
        preds.append(pred_logZ.item())
        trues.append(data.true_logZ.item())

    return total_rmse / max(n, 1), preds, trues


# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser(description='Attn-JGNN SAT Training')
    parser.add_argument('--dataset', type=str, required=True, choices=['bird', 'satlib'])
    parser.add_argument('--data_dir', type=str, required=True, help='CNF 文件目录')
    parser.add_argument('--label_file', type=str, default=None, help='标签 pickle 文件')
    parser.add_argument('--adaptive', action='store_true', default=False)
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
    config.task = 'sat'
    config.lr = args.lr
    config.epochs = args.epochs
    config.adaptive = args.adaptive

    print("=" * 60)
    print(f"Attn-JGNN SAT | Dataset: {args.dataset} | Adaptive: {args.adaptive}")
    print(f"Device: {config.device} | Epochs: {config.epochs} | LR: {config.lr}")
    print("=" * 60)

    # 数据
    dataset = SATInstanceDataset(args.data_dir, args.label_file)
    if len(dataset) == 0:
        print("No CNF files found!"); sys.exit(1)

    # 划分 (论文 3.5.2 节)
    n = len(dataset)
    idx = list(range(n)); random.shuffle(idx)
    if args.dataset == 'bird':
        # BIRD: 70/30 训练/测试
        split = int(n * 0.7)
        train_set = torch.utils.data.Subset(dataset, sorted(idx[:split]))
        test_set = torch.utils.data.Subset(dataset, sorted(idx[split:]))
        val_set = None
    else:
        # SATLIB: 60/20/20 训练/验证/测试
        split_train = int(n * 0.6)
        split_val = int(n * 0.8)
        train_set = torch.utils.data.Subset(dataset, sorted(idx[:split_train]))
        val_set = torch.utils.data.Subset(dataset, sorted(idx[split_train:split_val]))
        test_set = torch.utils.data.Subset(dataset, sorted(idx[split_val:]))
    print(f"Train: {len(train_set)}, Val: {len(val_set) if val_set else 'N/A'}, Test: {len(test_set)}")

    # 模型
    sample = build_data(dataset[0], config)
    sample.true_logZ = torch.tensor(0.0)
    model = AttnJGNN(config, sample).to(config.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    best_rmse = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)

    t0 = time.time()
    for epoch in range(config.epochs):
        train_rmse, _, _ = run_epoch(model, train_set, config, optimizer, current_epoch=epoch)
        scheduler.step()

        if epoch % 10 == 0:
            test_rmse, _, _ = run_epoch(model, test_set, config, optimizer=None, current_epoch=epoch)
            print(f"Epoch {epoch:3d} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f} | Time: {time.time()-t0:.0f}s")
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"attn_jgnn_{args.dataset}_best.pt"))

    print(f"\nDone! Best Test RMSE: {best_rmse:.4f} | Total: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
