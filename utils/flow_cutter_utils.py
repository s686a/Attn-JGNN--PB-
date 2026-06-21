import subprocess
import tempfile
import os
import platform
import networkx as nx
from typing import List
from config import Config


def _is_linux_binary(filepath):
    """检测文件是否为 Linux ELF 二进制"""
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(4)
        return magic[:4] == b'\x7fELF'
    except Exception:
        return False


def _to_wsl_path(win_path):
    """C:\\Users\\... -> /mnt/c/Users/..."""
    win_path = os.path.abspath(win_path).replace('\\', '/')
    if win_path[1] == ':':
        drive = win_path[0].lower()
        return f'/mnt/{drive}{win_path[2:]}'
    return win_path


def run_flow_cutter(graph: nx.Graph, timeout: int = 60,
                    flow_cutter_path: str = None,
                    target_treewidth: int = None) -> List[List[int]]:
    """
    调用 FlowCutter PACE，运行指定的时间（秒），
    然后从输出中解析最佳树分解的簇。

    参数:
        graph:            NetworkX 无向图
        timeout:          超时时间 (秒), 超时后发送 SIGINT
        flow_cutter_path: FlowCutter 可执行文件路径
        target_treewidth: 目标树宽 (需修改版 FlowCutter, 见 README §3.4)

    返回:
        clusters: 簇列表 (每个簇是节点索引的 list)
    """
    if flow_cutter_path is None:
        flow_cutter_path = Config.flow_cutter_path
    if not os.path.exists(flow_cutter_path):
        raise FileNotFoundError(f"FlowCutter executable not found at {flow_cutter_path}")

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    # 检测是否需要通过 WSL 调用 (Linux ELF 在 Windows 上)
    use_wsl = platform.system() == 'Windows' and _is_linux_binary(flow_cutter_path)
    if use_wsl:
        fc_wsl_path = _to_wsl_path(flow_cutter_path)

    # 写入临时 .gr 文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.gr', delete=False) as f:
        f.write(f"p tw {n_nodes} {n_edges}\n")
        for u, v in graph.edges():
            f.write(f"{u+1} {v+1}\n")
        temp_input = f.name

    temp_output = tempfile.NamedTemporaryFile(suffix='.td', delete=False)
    temp_output.close()

    # 构建命令
    if use_wsl:
        cmd = ['wsl', fc_wsl_path]
        input_arg = _to_wsl_path(temp_input)
    else:
        cmd = [flow_cutter_path]
        input_arg = temp_input

    if target_treewidth is not None:
        cmd += ['--target-width', str(target_treewidth)]
    cmd.append(input_arg)
    try:
        with open(temp_output.name, 'w') as out_f:
            subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        # 超时后，FlowCutter 会因 SIGINT 输出分解，但 subprocess.run 不会自动处理
        # 改用 Popen 手动发送 SIGINT
        proc = subprocess.Popen(cmd, stdout=open(temp_output.name, 'w'), stderr=subprocess.PIPE)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait()
    finally:
        pass

    # 解析输出
    clusters = []
    with open(temp_output.name, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('b'):
                parts = list(map(int, line.split()[1:]))
                if parts and parts[-1] == 0:
                    parts = parts[:-1]
                if parts:
                    clusters.append([p-1 for p in parts])

    os.unlink(temp_input)
    os.unlink(temp_output.name)
    return clusters