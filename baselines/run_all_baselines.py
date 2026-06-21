"""
统一基线实验编排脚本

依次运行:
  1. Attn-JGNN-PB (直接PB编码)
  2. PB→CNF(Warners)+Attn-JGNN
  3. PB→CNF(BDD)+Attn-JGNN
  4. PB→CNF(Warners)+NSNet
  5. PB→CNF(BDD)+NSNet
  6. ApproxMC-PB

用法:
    python baselines/run_all_baselines.py \
        --data_dir ./data \
        --attn_jgnn_checkpoint ./checkpoints/attn_jgnn_pb_best.pt \
        --nsnet_checkpoint ./checkpoints/nsnet_mc_best.pt \
        --approxmc_pb_path ./external/approxmc \
        --output_dir ./results
"""
import os, sys, argparse, json, time, glob, pickle, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_labels(data_dir):
    """加载 PB 标签"""
    label_file = os.path.join(data_dir, 'labels', 'pb_labels.pkl')
    if os.path.exists(label_file):
        with open(label_file, 'rb') as f:
            return pickle.load(f)
    return {}


def run_attn_jgnn_pb(pb_items, config, checkpoint_path, device):
    """Attn-JGNN-PB 直接 PB 编码"""
    from config import Config
    from models.attn_jgnn import AttnJGNN
    from graph.factor_graph import build_factor_graph
    from graph.join_graph import build_join_graph_adaptive

    if config is None:
        config = Config()
        config.task = 'pb'

    results = []
    sample_item = pb_items[0]
    sample_data = build_factor_graph(
        sample_item['num_vars'], sample_item['constraints'], task='pb')
    sample_data = build_join_graph_adaptive(
        sample_data, config.flow_cutter_path, task='pb')
    sample_data.true_logZ = torch.tensor(0.0)

    model = AttnJGNN(config, sample_data).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        sd = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(sd, strict=False)
    model.eval()

    t0 = time.time()
    for item in pb_items:
        data = build_factor_graph(
            item['num_vars'], item['constraints'], task='pb')
        data = build_join_graph_adaptive(
            data, config.flow_cutter_path, task='pb')
        data = data.to(device)
        with torch.no_grad():
            pred_logZ, _ = model(data)
        results.append({
            'file': item['filename'],
            'pred_logZ': float(pred_logZ.cpu().item()),
            'nv': item['num_vars']
        })
    elapsed = time.time() - t0
    return results, elapsed


def compute_summary(all_results, labels):
    """计算 RMSE 汇总"""
    summaries = {}
    for method_name, results in all_results.items():
        preds, trues = [], []
        for r in results:
            fn = r['file']
            true_count = labels.get(fn, 1)
            true_logZ = np.log(float(true_count))
            if r.get('pred_logZ') is not None:
                preds.append(r['pred_logZ'])
                trues.append(true_logZ)
        if preds:
            rmse = np.sqrt(np.mean(
                (np.array(preds) - np.array(trues)) ** 2))
            summaries[method_name] = {
                'rmse': float(rmse),
                'solved': len(preds),
                'avg_time': r.get('time', 0)
            }
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description='统一基线实验')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据根目录 (含 PB_Competition2006 等子目录)')
    parser.add_argument('--attn_jgnn_checkpoint', type=str, default=None,
                        help='Attn-JGNN-PB 检查点')
    parser.add_argument('--nsnet_checkpoint', type=str, default=None,
                        help='NSNet 检查点')
    parser.add_argument('--approxmc_pb_path', type=str,
                        default='./external/approxmc',
                        help='ApproxMC 可执行文件路径')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--timeout', type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # 加载 PB 测试数据
    from data.pb_parser import parse_opb, normalize_pb_constraints

    test_items = []
    for pname in ['PB_Competition2006', 'ExactCover', 'WSN']:
        dp = os.path.join(args.data_dir, pname, 'test')
        if not os.path.exists(dp):
            continue
        for fn in os.listdir(dp):
            if not (fn.endswith('.opb') or fn.endswith('.wpbf')):
                continue
            fp = os.path.join(dp, fn)
            try:
                nv, raw = parse_opb(fp, normalize=False)
                cs = normalize_pb_constraints(raw, normalize_coeffs=True)
                test_items.append({
                    'filepath': fp, 'filename': fn,
                    'num_vars': nv, 'constraints': cs
                })
            except Exception:
                pass

    labels = load_labels(args.data_dir)
    for item in test_items:
        item['true_count'] = labels.get(item['filename'], 1)

    print(f"Loaded {len(test_items)} test instances")
    print(f"Labels available: {len(labels)}")

    all_results = {}
    from config import Config

    # Attn-JGNN-PB (直接PB)
    if args.attn_jgnn_checkpoint:
        print("\n[1/4] Attn-JGNN-PB (直接PB编码)...")
        config = Config(); config.task = 'pb'
        results, elapsed = run_attn_jgnn_pb(
            test_items, config, args.attn_jgnn_checkpoint, device)
        all_results['Attn-JGNN-PB'] = {
            'results': results, 'time': elapsed,
            'solved': len(results)
        }
        print(f"  {len(results)} instances, {elapsed:.1f}s")

    # PB→CNF+Attn-JGNN (Warners & BDD)
    if args.attn_jgnn_checkpoint:
        from data.pb_to_cnf import pb_to_cnf_file
        from models.attn_jgnn import AttnJGNN
        from graph.factor_graph import build_factor_graph
        from graph.join_graph import build_join_graph_adaptive

        for enc_method in ['warners', 'bdd']:
            name = f'PB→CNF({enc_method})+Attn-JGNN'
            print(f"\n[ ] {name}...")
            config = Config(); config.task = 'sat'
            sample = build_factor_graph(2, [[1, 2]], task='sat')
            sample = build_join_graph_adaptive(
                sample, config.flow_cutter_path, task='sat')
            sample.true_logZ = torch.tensor(0.0)
            model = AttnJGNN(config, sample).to(device)
            if os.path.exists(args.attn_jgnn_checkpoint):
                ckpt = torch.load(args.attn_jgnn_checkpoint,
                                  map_location=device)
                sd = ckpt.get('model_state_dict', ckpt)
                model.load_state_dict(sd, strict=False)
            model.eval()

            results = []
            t0 = time.time()
            with tempfile.TemporaryDirectory() as td:
                for item in test_items:
                    try:
                        cnf_p = os.path.join(
                            td, f"{item['filename']}.{enc_method}.cnf")
                        pb_to_cnf_file(
                            item['constraints'], item['num_vars'],
                            cnf_p, method=enc_method)
                        # 重新解析CNF用于Attn-JGNN
                        from data.cnf_parser import parse_cnf
                        nv_cnf, clauses_cnf = parse_cnf(cnf_p)
                        if nv_cnf == 0 or not clauses_cnf:
                            continue
                        data = build_factor_graph(
                            nv_cnf, clauses_cnf, task='sat')
                        data = build_join_graph_adaptive(
                            data, config.flow_cutter_path, task='sat')
                        data = data.to(device)
                        with torch.no_grad():
                            pred_logZ, _ = model(data)
                        results.append({
                            'file': item['filename'],
                            'pred_logZ': float(pred_logZ.cpu().item()),
                            'nv_orig': item['num_vars'],
                            'nv_cnf': nv_cnf
                        })
                    except Exception as e:
                        pass
            elapsed = time.time() - t0
            all_results[name] = {
                'results': results, 'time': elapsed,
                'solved': len(results)
            }
            print(f"  {len(results)} instances, {elapsed:.1f}s")

    # NSNet baselines
    if args.nsnet_checkpoint:
        NSNET_SRC = r"C:\Users\123456\Desktop\NSNet-main\NSNet-main\src"
        if NSNET_SRC not in sys.path:
            sys.path.insert(0, NSNET_SRC)
        # 使用本地复制的 NSNet 模型
        from baselines.nsnet_runner import NSNetMCRunner
        nsnet_runner = NSNetMCRunner(args.nsnet_checkpoint)

        for enc_method in ['warners', 'bdd']:
            name = f'PB→CNF({enc_method})+NSNet'
            print(f"\n[ ] {name}...")
            results = []
            t0 = time.time()
            with tempfile.TemporaryDirectory() as td:
                for item in test_items:
                    try:
                        cnf_p = os.path.join(
                            td, f"{item['filename']}.{enc_method}.cnf")
                        from data.pb_to_cnf import pb_to_cnf_file
                        pb_to_cnf_file(
                            item['constraints'], item['num_vars'],
                            cnf_p, method=enc_method)
                        plz = nsnet_runner.predict_logz(cnf_p)
                        if plz is not None:
                            results.append({
                                'file': item['filename'],
                                'pred_logZ': float(plz),
                                'nv_orig': item['num_vars']
                            })
                    except Exception:
                        pass
            elapsed = time.time() - t0
            all_results[name] = {
                'results': results, 'time': elapsed,
                'solved': len(results)
            }
            print(f"  {len(results)} instances, {elapsed:.1f}s")

    # ApproxMC-PB
    approxmc_path = args.approxmc_pb_path
    if approxmc_path and os.path.exists(approxmc_path):
        print("\n[6] ApproxMC-PB...")
        from baselines.approxmc_pb_runner import ApproxMC_PB_Runner
        amc_runner = ApproxMC_PB_Runner(
            approxmc_path, timeout=args.timeout)
        results = []
        t0 = time.time()
        for item in test_items:
            try:
                success, logZ, t = amc_runner.run(item['filepath'])
                if success:
                    results.append({
                        'file': item['filename'],
                        'pred_logZ': logZ,
                        'nv': item['num_vars'],
                        'time': t
                    })
            except Exception:
                pass
        elapsed = time.time() - t0
        all_results['ApproxMC-PB'] = {
            'results': results, 'time': elapsed,
            'solved': len(results)
        }
        print(f"  {len(results)} instances, {elapsed:.1f}s")

    # 汇总对比
    print(f"\n{'=' * 80}")
    print("RMSE 对比 (对数计数)")
    print('=' * 80)
    print(f"{'方法':<35} {'Solved':<8} {'RMSE':<10} {'Time(s)':<10}")
    print('-' * 65)

    for method_name, result_data in all_results.items():
        preds, trues = [], []
        for r in result_data['results']:
            fn = r['file']
            tc = labels.get(fn, 1)
            tz = np.log(float(tc))
            if r.get('pred_logZ') is not None:
                preds.append(r['pred_logZ'])
                trues.append(tz)
        if preds:
            rmse = np.sqrt(np.mean(
                (np.array(preds) - np.array(trues)) ** 2))
        else:
            rmse = float('nan')
        solved = result_data.get('solved', len(result_data['results']))
        total_time = result_data.get('time', 0)
        print(f"{method_name:<35} {solved:<8} {rmse:<10.4f} "
              f"{total_time:<10.1f}")

    # 保存
    output_file = os.path.join(args.output_dir, 'all_baselines.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {output_file}")


if __name__ == '__main__':
    main()