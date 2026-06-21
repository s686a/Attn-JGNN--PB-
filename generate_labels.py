"""
真值标签生成工具

为 SAT 数据集调用 DSharp 生成真值模型计数。
为 PB 数据集调用 PBMC 或 PBCount 生成真值。

所有求解器均为 Linux 二进制, 需通过 WSL 调用 (加 --wsl 标志)。

用法:
    # SAT: 使用 DSharp (WSL)
    python generate_labels.py --solver dsharp --data_dir ./data/SATLIB \
        --solver_path ~/solvers/dsharp/dsharp --output ./data/satlib_labels.pkl --wsl

    # PB: 使用 PBMC (WSL)
    python generate_labels.py --solver pbmc --data_dir ./data/PB_Competition2006 \
        --solver_path ~/solvers/pbmc/bin/pbmc --output ./data/pb_labels.pkl --wsl

    # PB: 使用 PBCount (WSL, 交叉验证)
    python generate_labels.py --solver pbcount --data_dir ./data/PB_Competition2006 \
        --solver_path ~/solvers/pbcount/pbcount --output ./data/pb_labels.pkl --wsl

    # 也可用 Windows .bat 包装脚本 (external/ 目录下)
    python generate_labels.py --solver dsharp --data_dir ./data/SATLIB \
        --solver_path ./external/dsharp.bat --output ./data/satlib_labels.pkl
"""
import os, sys, argparse, glob, pickle, subprocess, time, signal, re
import numpy as np


def run_dsharp(cnf_path, dsharp_path, timeout=5000):
    """调用 DSharp 精确计数（论文 3.5.2 节: 时间限制 5000s）"""
    cmd = [dsharp_path, cnf_path, '-F', '-cnf', '-noPP']
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        stdout, stderr = proc.communicate(timeout=timeout)
        for line in stdout.split('\n'):
            if '#SAT' in line and 'full' in line.lower():
                parts = line.strip().split()
                for p in parts:
                    try:
                        return int(p)
                    except ValueError:
                        continue
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except:
            pass
    return None


def run_pbcount(opb_path, pbcount_path, timeout=3600, weighted=False):
    """调用 PBCount 精确计数（AAAI 2024/2025）

    PBCount 输出格式:
        s mc 5.000000
    """
    wf = '2' if weighted else '1'
    cmd = [pbcount_path, '--cf', opb_path, '--wf', wf]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        stdout, stderr = proc.communicate(timeout=timeout)
        # PBCount 输出 "s mc <count>"
        for line in stdout.split('\n'):
            line_stripped = line.strip()
            if line_stripped.startswith('s mc'):
                parts = line_stripped.split()
                if len(parts) >= 3:
                    try:
                        return float(parts[2])
                    except ValueError:
                        continue
        # 备选：向下兼容旧格式
        for line in stdout.split('\n'):
            if 'mc' in line:
                match = re.search(r'mc\s+([\d.]+)', line)
                if match:
                    return float(match.group(1))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except:
            pass
    return None


def run_pbmc(opb_path, pbmc_path, timeout=3600):
    """调用 PBMC 精确计数"""
    cmd = [pbmc_path, opb_path]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        stdout, stderr = proc.communicate(timeout=timeout)
        for line in stdout.split('\n'):
            if 'count' in line.lower() or 'solution' in line.lower():
                nums = re.findall(r'\d+', line)
                if nums:
                    return int(nums[-1])
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except:
            pass
    return None


# ============================================================
# WSL 版本 — 在 Windows 上通过 WSL 调用 Linux 求解器
# ============================================================

def to_wsl_path(win_path):
    """C:\\Users\\... -> /mnt/c/Users/..."""
    win_path = os.path.abspath(win_path)
    drive = win_path[0].lower()
    rest = win_path[2:].replace('\\', '/')
    return f'/mnt/{drive}{rest}'


def run_dsharp_wsl(cnf_path, dsharp_linux_path, timeout=5000):
    """通过 WSL 调用 DSharp"""
    cmd = ['wsl', dsharp_linux_path, to_wsl_path(cnf_path), '-F', '-cnf', '-noPP']
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        stdout, stderr = proc.communicate(timeout=timeout)
        for line in stdout.split('\n'):
            if '#SAT' in line and 'full' in line.lower():
                parts = line.strip().split()
                for p in parts:
                    try:
                        return int(p)
                    except ValueError:
                        continue
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except:
            pass
    return None


def run_pbmc_wsl(opb_path, pbmc_linux_path, timeout=3600):
    """通过 WSL 调用 PBMC"""
    cmd = ['wsl', pbmc_linux_path, to_wsl_path(opb_path)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        stdout, stderr = proc.communicate(timeout=timeout)
        for line in stdout.split('\n'):
            if 'count' in line.lower() or 'solution' in line.lower():
                nums = re.findall(r'\d+', line)
                if nums:
                    return int(nums[-1])
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except:
            pass
    return None


def run_pbcount_wsl(opb_path, pbcount_linux_path, timeout=3600):
    """通过 WSL 调用 PBCount"""
    wf = '1'  # unweighted
    cmd = ['wsl', pbcount_linux_path, '--cf', to_wsl_path(opb_path), '--wf', wf]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        stdout, stderr = proc.communicate(timeout=timeout)
        for line in stdout.split('\n'):
            if line.strip().startswith('s mc'):
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        return float(parts[2])
                    except ValueError:
                        continue
        for line in stdout.split('\n'):
            if 'mc' in line:
                match = re.search(r'mc\s+([\d.]+)', line)
                if match:
                    return float(match.group(1))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description='Generate ground truth labels')
    parser.add_argument('--solver', type=str, required=True,
                        choices=['dsharp', 'pbmc', 'pbcount'],
                        help='Solver: dsharp (SAT), pbmc (PB), pbcount (PB, AAAI2024)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing .cnf or .opb files')
    parser.add_argument('--solver_path', type=str, required=True,
                        help='Path to solver executable')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle file')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Solver timeout in seconds')
    parser.add_argument('--weighted', action='store_true',
                        help='Use weighted model counting (only for pbcount)')
    parser.add_argument('--wsl', action='store_true',
                        help='Run solver via WSL (Windows Subsystem for Linux)')
    args = parser.parse_args()

    def to_wsl_path(win_path):
        """将 Windows 路径转为 WSL 路径: C:\\... -> /mnt/c/..."""
        win_path = os.path.abspath(win_path)
        drive = win_path[0].lower()
        rest = win_path[2:].replace('\\', '/')
        return f'/mnt/{drive}{rest}'

    def make_wsl_cmd(solver_linux_path, file_path):
        """构建 WSL 命令: wsl <solver> <wsl_file_path>"""
        return ['wsl', solver_linux_path, to_wsl_path(file_path)]

    # 根据 solver 选择文件类型和求解函数
    if args.solver == 'dsharp':
        timeout = args.timeout or 5000
        files = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.cnf'), recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(args.data_dir, '*.cnf')))
        if args.wsl:
            run_solver = lambda fp: run_dsharp_wsl(fp, args.solver_path, timeout)
        else:
            run_solver = lambda fp: run_dsharp(fp, args.solver_path, timeout)

    elif args.solver == 'pbcount':
        timeout = args.timeout or 3600
        files = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.opb'), recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.wpbf'), recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.pb'), recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(args.data_dir, '*.opb')))
        if not files:
            files = sorted(glob.glob(os.path.join(args.data_dir, '*.wpbf')))
        if args.wsl:
            run_solver = lambda fp: run_pbcount_wsl(fp, args.solver_path, timeout)
        else:
            run_solver = lambda fp: run_pbcount(fp, args.solver_path, timeout, args.weighted)

    else:  # pbmc
        timeout = args.timeout or 3600
        files = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.opb'), recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.wpbf'), recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.pb'), recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(args.data_dir, '*.opb')))
        if not files:
            files = sorted(glob.glob(os.path.join(args.data_dir, '*.wpbf')))
        if args.wsl:
            run_solver = lambda fp: run_pbmc_wsl(fp, args.solver_path, timeout)
        else:
            run_solver = lambda fp: run_pbmc(fp, args.solver_path, timeout)

    if not files:
        print(f"No input files found in {args.data_dir}!")
        print("  For DSharp: expects .cnf files")
        print("  For PBMC/PBCount: expects .opb or .pb files")
        sys.exit(1)

    print(f"Solver: {args.solver}")
    print(f"Data dir: {args.data_dir}")
    print(f"Generating labels for {len(files)} files (timeout={timeout}s)")
    print("-" * 60)

    labels = {}
    solved, failed = 0, 0
    for i, fp in enumerate(files):
        fn = os.path.basename(fp)
        t0 = time.time()
        count = run_solver(fp)
        elapsed = time.time() - t0

        if count is not None:
            labels[fn] = count
            solved += 1
            if isinstance(count, float) and count == int(count):
                print(f"  [{i+1}/{len(files)}] {fn}: {int(count):,} ({elapsed:.1f}s)")
            else:
                print(f"  [{i+1}/{len(files)}] {fn}: {count:,} ({elapsed:.1f}s)")
        else:
            failed += 1
            print(f"  [{i+1}/{len(files)}] {fn}: FAILED ({elapsed:.1f}s)")

    with open(args.output, 'wb') as f:
        pickle.dump(labels, f)

    print("-" * 60)
    print(f"Done! Solved: {solved}/{len(files)} | Failed: {failed}/{len(files)}")
    print(f"Labels saved: {args.output}")


if __name__ == '__main__':
    main()
