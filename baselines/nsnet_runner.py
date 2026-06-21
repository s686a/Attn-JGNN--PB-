"""
NSNet 模型计数
将 PB 公式转换为 CNF 后调用 NSNet 进行模型计数估计。
支持 Warners 和 BDD 两种编码。

用法:
    python baselines/nsnet_runner.py --pb_data ./data/ExactCover/test \
        --encoding warners --nsnet_checkpoint ./checkpoints/nsnet_mc_best.pt \
        --output ./results/nsnet_warners.json
"""
import os, sys, argparse, json, time, tempfile, glob, itertools
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSNET_SRC = r"C:\Users\123456\Desktop\NSNet-main\NSNet-main\src"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, NSNET_SRC)  

from data.pb_parser import parse_opb, normalize_pb_constraints
from data.pb_to_cnf import pb_to_cnf_file


class NSNetMCRunner:
    """对转换后的 CNF 用 NSNet 推理 logZ"""

    def __init__(self, checkpoint_path):
        # 使用本地复制的 NSNet 模型 (避免与 Attn-JGNN models/ 冲突)
        from baselines.nsnet_model.nsnet import NSNet
        # NSNet 的 options 仍需从原始路径导入
        from utils.options import add_model_options
        p = argparse.ArgumentParser()
        add_model_options(p)
        opts = p.parse_args([])
        opts.task = 'model-counting'
        opts.dim = 64
        opts.n_rounds = 10
        opts.n_mlp_layers = 3
        opts.activation = 'relu'
        opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.opts = opts
        self.device = opts.device
        self.model = NSNet(opts).to(opts.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=opts.device)
            sd = ckpt.get('state_dict', ckpt)
            self.model.load_state_dict(sd, strict=False)
        self.model.eval()

    def _cnf_to_bpg(self, cnf_path):
        """CNF文件 -> NSNet BPG 图格式"""
        from utils.utils import parse_cnf_file
        from utils.dataset import BPG
        nv, clauses = parse_cnf_file(cnf_path)
        if nv == 0:
            return None

        sign_l, type_e = [], []
        c2l_mc = {l: [] for l in range(2 * nv)}
        c2l_me = {l: [] for l in range(2 * nv)}
        c2l_rl, c2l_sl = [], []
        l2c_rl, l2c_sal, l2c_sl = [], [], []
        cbr, cbs, cbn = [], [], []
        vd = torch.zeros(nv)
        ib = 0
        mgi = 0
        for ci, cl in enumerate(clauses):
            uv = sorted(set(abs(l) - 1 for l in cl))
            for m, v in enumerate(uv):
                pl, nl = v * 2, v * 2 + 1
                pm, nm = ib + m * 2, ib + m * 2 + 1
                sign_l.extend([pl, nl])
                c2l_mc[pl].append(ci)
                c2l_mc[nl].append(ci)
                c2l_me[pl].append(pm)
                c2l_me[nl].append(nm)
                type_e.append(1 if (v + 1) in cl else 0)
                type_e.append(1 if -(v + 1) in cl else 0)
            for sd, dv in enumerate(uv):
                for idxs in np.ndindex(tuple([2] * (len(uv) - 1))):
                    mt = [(ib + m * 2 + 1, ib + m * 2)
                          for m, v2 in enumerate(uv) if v2 != dv]
                    ass = [type_e[mt[i][idx]] for i, idx in enumerate(idxs)]
                    sat = sum(ass) > 0
                    rpt = [mt[i][idx] for i, idx in enumerate(idxs)]
                    if type_e[ib + sd * 2] or sat:
                        l2c_rl.append(rpt)
                        l2c_sal.append([mgi] * len(rpt))
                        mgi += 1
                        l2c_sl.append(ib + sd * 2)
                    if type_e[ib + sd * 2 + 1] or sat:
                        l2c_rl.append(rpt)
                        l2c_sal.append([mgi] * len(rpt))
                        mgi += 1
                        l2c_sl.append(ib + sd * 2 + 1)
            ib += len(uv) * 2

        ib = 0
        for ci, cl in enumerate(clauses):
            uv = sorted(set(abs(l) - 1 for l in cl))
            for m, v in enumerate(uv):
                for p in [v * 2, v * 2 + 1]:
                    pm = ib + m * 2 + (0 if p == v * 2 else 1)
                    for nc, ne in zip(c2l_mc[p], c2l_me[p]):
                        if nc != ci:
                            c2l_rl.append(ne)
                            c2l_sl.append(pm)
            ib += len(uv) * 2

        ib = 0
        bfi = 0
        for ci, cl in enumerate(clauses):
            uv = set(abs(l) - 1 for l in cl)
            for idxs in np.ndindex(tuple([2] * len(uv))):
                mt = [(ib + m * 2 + 1, ib + m * 2)
                      for m, v in enumerate(uv)]
                ass = [type_e[mt[i][idx]] for i, idx in enumerate(idxs)]
                if sum(ass) > 0:
                    rpt = [mt[i][idx] for i, idx in enumerate(idxs)]
                    cbr.append(rpt)
                    cbs.append([bfi] * len(rpt))
                    bfi += 1
                    cbn.append(ci)
            for v in uv:
                vd[v] += 1
            ib += len(uv) * 2

        data = BPG(
            l_size=torch.tensor([nv * 2]),
            c_size=torch.tensor([len(clauses)]),
            sign_l_edge_index=torch.tensor(sign_l, dtype=torch.long),
            c2l_msg_repeat_index=torch.tensor(c2l_rl, dtype=torch.long),
            c2l_msg_scatter_index=torch.tensor(c2l_sl, dtype=torch.long),
            l2c_msg_aggr_repeat_index=torch.tensor(
                list(itertools.chain(*l2c_rl)), dtype=torch.long),
            l2c_msg_aggr_scatter_index=torch.tensor(
                list(itertools.chain(*l2c_sal)), dtype=torch.long),
            l2c_msg_scatter_index=torch.tensor(l2c_sl, dtype=torch.long),
            c_blf_repeat_index=torch.tensor(
                list(itertools.chain(*cbr)), dtype=torch.long),
            c_blf_scatter_index=torch.tensor(
                list(itertools.chain(*cbs)), dtype=torch.long),
            c_blf_norm_index=torch.tensor(cbn, dtype=torch.long),
            v_degrees=vd,
            c_batch=torch.zeros(len(clauses), dtype=torch.long),
            v_batch=torch.zeros(nv, dtype=torch.long),
        )
        return data

    def predict_logz(self, cnf_path):
        """对单个 CNF 推理, 返回 logZ"""
        data = self._cnf_to_bpg(cnf_path)
        if data is None:
            return None
        data = data.to(self.device)
        with torch.no_grad():
            logZ = self.model(data)
        return logZ.cpu().item()


def main():
    parser = argparse.ArgumentParser(description='NSNet PB 基线运行器')
    parser.add_argument('--pb_data', type=str, required=True,
                        help='PB 数据目录')
    parser.add_argument('--encoding', type=str,
                        choices=['warners', 'bdd', 'both'],
                        default='warners')
    parser.add_argument('--nsnet_checkpoint', type=str, required=True,
                        help='NSNet 模型检查点 .pt')
    parser.add_argument('--output', type=str,
                        default='./results/nsnet_pb.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    pb_files = sorted(glob.glob(os.path.join(args.pb_data, '*.wpbf')))
    if not pb_files:
        pb_files = sorted(glob.glob(os.path.join(args.pb_data, '*.opb')))
    if not pb_files:
        pb_files = sorted(glob.glob(
            os.path.join(args.pb_data, '**', '*.wpbf'), recursive=True))
    if not pb_files:
        pb_files = sorted(glob.glob(
            os.path.join(args.pb_data, '**', '*.opb'), recursive=True))
    print(f"Found {len(pb_files)} PB files")

    runner = NSNetMCRunner(args.nsnet_checkpoint)
    methods = (['warners', 'bdd'] if args.encoding == 'both'
               else [args.encoding])
    all_results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for method in methods:
            print(f"\n{'=' * 60}")
            print(f"Encoding: {method}")
            print('=' * 60)
            results = []
            t0 = time.time()
            solved = 0
            for fi, fp in enumerate(pb_files):
                fn = os.path.basename(fp)
                try:
                    nv, raw = parse_opb(fp, normalize=False)
                    cs = normalize_pb_constraints(raw,
                                                  normalize_coeffs=True)
                    cnf_path = os.path.join(tmpdir,
                                            f'{fn}.{method}.cnf')
                    nnv, nnc = pb_to_cnf_file(cs, nv, cnf_path,
                                              method=method)
                    plz = runner.predict_logz(cnf_path)
                    if plz is not None:
                        results.append({
                            'file': fn,
                            'nv_orig': nv,
                            'nv_cnf': nnv,
                            'nc_cnf': nnc,
                            'pred_logZ': float(plz),
                            'pred_count': float(np.exp(min(plz, 50.0)))
                        })
                        solved += 1
                except Exception as e:
                    print(f"  [{fi + 1}] {fn}: ERROR - {e}")
                if (fi + 1) % 20 == 0:
                    print(f"  [{fi + 1}/{len(pb_files)}] "
                          f"{method}: {solved} solved")

            elapsed = time.time() - t0
            all_results[method] = {
                'results': results,
                'time': elapsed,
                'solved': solved
            }
            print(f"  {method}: {solved}/{len(pb_files)} "
                  f"in {elapsed:.1f}s")

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()