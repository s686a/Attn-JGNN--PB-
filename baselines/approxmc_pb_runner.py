"""
ApproxMC-PB 近似计数

用法:
    python baselines/approxmc_pb_runner.py \
        --pb_data ./data/PB_Competition2006/test \
        --approxmc_pb_path ./external/approxmc-pb \
        --output ./results/approxmc_pb.json --timeout 5000

如果需要下载 ApproxMC-PB:
    git clone https://github.com/meelgroup/approxmc
    cd approxmc && git submodule update --init
    或从 https://github.com/meelgroup/arjun 获取 PB 支持版本
"""
import os, sys, argparse, json, time, glob, subprocess, signal, re, tempfile
import numpy as np


class ApproxMC_PB_Runner:
    """
    ApproxMC-PB 求解器封装。
    ApproxMC 本身不直接支持 .opb 格式 —
    需要将 PB 约束转换为 CNF 后调用 ApproxMC，
    或使用支持 PB-XOR 的特殊版本。
    """

    def __init__(self, solver_path, timeout=5000, seed=42):
        self.solver_path = solver_path
        self.timeout = timeout
        self.seed = seed

    def run(self, opb_path):
        """
        对 .opb 文件运行 ApproxMC-PB。
        返回: (success, log_count, elapsed_time)
        如果未找到原生 PB 支持, 回退到: PB→CNF(Warners)→ApproxMC。
        """
        t0 = time.time()

        # 方案 A: 尝试直接调用 (如果 ApproxMC 支持 PB)
        if self._check_pb_support():
            return self._run_direct(opb_path)
        # 方案 B: PB→CNF 回退
        return self._run_via_cnf(opb_path)

    def _check_pb_support(self):
        """检查 ApproxMC 是否原生支持 PB 格式"""
        try:
            result = subprocess.run(
                [self.solver_path, '--help'],
                capture_output=True, text=True, timeout=10)
            return 'opb' in result.stdout.lower() or 'pb' in result.stdout.lower()
        except Exception:
            return False

    def _run_direct(self, opb_path):
        """直接对 .opb 文件运行 ApproxMC"""
        t0 = time.time()
        cmd = [self.solver_path, '--seed', str(self.seed)]
        cmd.append(opb_path)

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=True)
            stdout, _ = proc.communicate(timeout=self.timeout)

            # 解析输出: 查找 "Number of solutions is: N" 或类似模式
            count = None
            for pattern in [
                r'Number of solutions is:\s*(\d+)\s*\n',
                r'# solutions\s*[:=]\s*(\d+)',
                r'c count\s+(\d+)',
                r's mc\s+(\d+)',
            ]:
                match = re.search(pattern, stdout)
                if match:
                    count = int(match.group(1))
                    break

            if count is not None:
                return True, float(np.log(max(count, 1))), time.time() - t0
            else:
                return False, -1, time.time() - t0

        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            return False, -1, self.timeout

    def _run_via_cnf(self, opb_path):
        """回退方案: PB→CNF(Warners)→ApproxMC"""
        from data.pb_parser import parse_opb, normalize_pb_constraints
        from data.pb_to_cnf import pb_to_cnf_file

        t0 = time.time()

        try:
            nv, raw = parse_opb(opb_path, normalize=False)
            constraints = normalize_pb_constraints(raw, normalize_coeffs=True)
        except Exception:
            return False, -1, time.time() - t0

        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.cnf', delete=False) as f:
            cnf_path = f.name
        try:
            pb_to_cnf_file(constraints, nv, cnf_path, method='warners')

            # 调用 ApproxMC (CNF 模式)
            cmd = [self.solver_path, '--seed', str(self.seed)]
            cmd.append(cnf_path)

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=True)
            stdout, _ = proc.communicate(timeout=self.timeout)

            count = None
            for pattern in [
                r'Number of solutions is:\s*(\d+)\s*\n',
                r'# solutions\s*[:=]\s*(\d+)',
                r'c (?:s|count)\s+(\d+)',
                r's mc\s+(\d+)',
            ]:
                match = re.search(pattern, stdout)
                if match:
                    count = int(match.group(1))
                    break

            if count is not None:
                return True, float(np.log(max(count, 1))), time.time() - t0
            else:
                return False, -1, time.time() - t0

        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            return False, -1, self.timeout
        finally:
            if os.path.exists(cnf_path):
                os.unlink(cnf_path)


def main():
    parser = argparse.ArgumentParser(description='ApproxMC-PB 基线运行器')
    parser.add_argument('--pb_data', type=str, required=True,
                        help='PB 数据目录 (.opb 文件)')
    parser.add_argument('--approxmc_pb_path', type=str,
                        default='./external/approxmc',
                        help='ApproxMC 可执行文件路径')
    parser.add_argument('--output', type=str,
                        default='./results/approxmc_pb.json')
    parser.add_argument('--timeout', type=int, default=5000,
                        help='每个实例超时时间 (秒)')
    parser.add_argument('--seed', type=int, default=42)
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

    runner = ApproxMC_PB_Runner(args.approxmc_pb_path,
                                timeout=args.timeout, seed=args.seed)

    results = []
    solved = 0
    total_time = 0

    for fi, fp in enumerate(pb_files):
        fn = os.path.basename(fp)
        print(f"  [{fi + 1}/{len(pb_files)}] {fn} ...", end=' ', flush=True)

        success, log_count, elapsed = runner.run(fp)
        total_time += elapsed

        if success:
            solved += 1
            results.append({
                'file': fn,
                'pred_logZ': log_count,
                'pred_count': float(np.exp(min(log_count, 700))),
                'time': elapsed
            })
            print(f"OK (logZ={log_count:.2f}, {elapsed:.1f}s)")
        else:
            results.append({
                'file': fn,
                'pred_logZ': None,
                'time': elapsed,
                'status': 'timeout' if elapsed >= args.timeout else 'failed'
            })
            print(f"FAIL/Timeout ({elapsed:.1f}s)")

    summary = {
        'solver': 'ApproxMC-PB',
        'timeout': args.timeout,
        'total_instances': len(pb_files),
        'solved': solved,
        'total_time': total_time,
        'avg_time': total_time / max(solved, 1),
        'results': results
    }

    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"ApproxMC-PB: {solved}/{len(pb_files)} solved")
    print(f"Total time: {total_time:.1f}s, "
          f"Avg: {total_time / max(solved, 1):.1f}s/instance")
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()