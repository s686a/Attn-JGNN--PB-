import os
import re
import signal
import time
import subprocess
import numpy as np
import pickle
import torch
from decimal import Decimal
from models.attn_jgnn import AttnJGNN
from config import get_default_config
from flow_cutter_utils import FlowCutterDataProcessor

class SATSolver:
    def __init__(self, opts):
        self.opts = opts
        if opts.solver == 'CaDiCaL':
            self.exec_dir = os.path.abspath('external/CaDiCaL')
            self.cmd_line = ['./cadical']
        elif opts.solver == 'Sparrow':
            self.exec_dir = os.path.abspath('external/Sparrow')
            self.cmd_line = ['./sparrow', '-a', '-l', '-r1']
            if opts.max_flips is not None:
                self.cmd_line.append('--maxflips')
                self.cmd_line.append(str(opts.max_flips))

    def run(self, input_filepath):
        filename = os.path.splitext(os.path.basename(input_filepath))[0]        
        cmd_line = self.cmd_line.copy()
        if self.opts.solver == 'Sparrow' and self.opts.model is not None:
            tmp_filepath = os.path.join(os.path.dirname(input_filepath), filename + '_' + self.opts.solver + '_' + self.opts.model + '.out')
            init_filepath = os.path.join(os.path.dirname(input_filepath), filename + '_' + self.opts.model + '.out')
            cmd_line.append('-f')
            cmd_line.append(init_filepath)
        else:
            tmp_filepath = os.path.join(os.path.dirname(input_filepath), filename + '_' + self.opts.solver + '.out')
        
        cmd_line.append(input_filepath)

        with open(tmp_filepath, 'w') as f:
            t0 = time.time()
            timeout_expired = 0
            try:
                process = subprocess.Popen(cmd_line, stdout=f, stderr=f, cwd=self.exec_dir, start_new_session=True)
                process.communicate(timeout=self.opts.timeout)
            except:
                timeout_expired = 1
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            t = time.time() - t0
        
        complete = 0
        assignment = []
        num_flips = 0

        if timeout_expired or os.stat(tmp_filepath).st_size == 0:
            os.remove(tmp_filepath)
            return complete, assignment, num_flips, t
        
        with open(tmp_filepath, 'r') as f:
            for line in f.readlines():
                if line.startswith('v'):
                    assignment = assignment + [int(s) for s in line.strip().split()[1:]]
                if line.startswith('c numFlips'):
                    num_flips = Decimal(line.strip().split()[-1])
        
        if assignment:
            complete = 1
            assignment = np.array(assignment[:-1]) > 0
        
        os.remove(tmp_filepath)
        return complete, assignment, num_flips, t


class MCSolver:
    def __init__(self, opts):
        self.opts = opts
        self.aein_model = None  
        self.data_processor = None  
        
        if opts.solver == 'DSHARP':
            self.exec_dir = os.path.abspath('external/DSHARP')
            self.cmd_line = ['./dsharp']
            self.cnt_pattern = '#SAT \(full\):   \t\t(.+)\n'
        elif opts.solver == 'ApproxMC3':
            self.exec_dir = os.path.abspath('external/ApproxMC3')
            self.cmd_line = ['./approxmc3']
            self.cnt_pattern = 'Number of solutions is: (.+)\n'
        elif opts.solver == 'F2':
            self.exec_dir = os.path.abspath('external/F2')
            self.cmd_line = ['python', 'f2.py', '--random-seed', str(abs(opts.seed)+1), 
                            '--sharpsat-exe', 'sharpsat', '--mode', 'lb', 
                            '--max-time', str(opts.timeout), '--skip-sharpsat']
            self.cnt_pattern = 'F2: Lower bound is (.+) \('
        elif opts.solver == 'AEIN':
            self.exec_dir = os.path.abspath('models')  
            self._init_attn_jgnn() 
        else:
            raise ValueError(f"不支持的求解器: {opts.solver}")

    def _init_attn_jgnn(self):
        """加载Attn-JGNN模型 (注意: 需要先构建data对象来初始化模型参数)"""
        from config import Config
        from graph.factor_graph import build_factor_graph
        from graph.join_graph import build_join_graph_adaptive

        self.config = Config()
        self.config.device = self.opts.device if hasattr(self.opts, 'device') else 'cuda'
        self.config.task = 'sat'

        # 先构建一个占位 data 对象来初始化模型 (模型需要 data 来确定 edge_feat_dim)
        dummy_clauses = [[1, 2]]
        dummy_data = build_factor_graph(2, dummy_clauses, task='sat')
        self.attn_jgnn = AttnJGNN(self.config, dummy_data).to(self.config.device)

        model_path = os.path.join(self.exec_dir, self.opts.aein_model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Attn-JGNN model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.config.device)
        if 'model_state_dict' in checkpoint:
            self.attn_jgnn.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.attn_jgnn.load_state_dict(checkpoint)
        self.attn_jgnn.eval()

    def _run_attn_jgnn(self, input_filepath):
        """
        使用 Attn-JGNN 进行推理。
        注意: 此方法需要输入为 CNF 文件路径 (DIMACS 格式)。
        """
        from data.cnf_parser import parse_cnf
        from graph.factor_graph import build_factor_graph
        from graph.join_graph import build_join_graph_adaptive
        import numpy as np

        t0 = time.time()
        try:
            nv, clauses = parse_cnf(input_filepath)
            data = build_factor_graph(nv, clauses, task='sat')
            data = build_join_graph_adaptive(data, self.config.flow_cutter_path, task='sat')
            data = data.to(self.config.device)

            with torch.no_grad():
                pred_logZ, _ = self.attn_jgnn(data)
            counting = Decimal(float(np.exp(pred_logZ.cpu().item())))
            complete = 1
        except Exception as e:
            print(f"Attn-JGNN inference error: {e}")
            counting = Decimal(-1)
            complete = 0

        t = time.time() - t0
        return complete, counting, t

    def run(self, input_filepath):
        if self.opts.solver == 'AEIN':
            return self._run_aein(input_filepath)
        
        filename = os.path.splitext(os.path.basename(input_filepath))[0]
        cmd_line = self.cmd_line.copy()
        cmd_line.append(input_filepath)
        stdout = ''

        t0 = time.time()
        timeout_expired = 0
        try:
            process = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, 
                                      cwd=self.exec_dir, text=True, start_new_session=True)
            stdout, _ = process.communicate(timeout=self.opts.timeout)
        except:
            timeout_expired = 1
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        t = time.time() - t0
        
        complete = 0
        counting = -1

        matches = re.search(self.cnt_pattern, stdout)

        if self.opts.solver == 'F2':
            all_files = os.listdir(self.exec_dir)
            for tmp_file in all_files:
                if tmp_file.endswith('.cnf') and filename in tmp_file:
                    os.remove(os.path.join(self.exec_dir, tmp_file))

        if timeout_expired or not matches:
            return complete, counting, t

        complete = 1
        if 'x' not in matches[1] and '^' not in matches[1]:
            counting = Decimal(matches[1])
            if Decimal.is_nan(counting):
                complete = 0
        else:
            counting = Decimal(eval(matches[1].replace('x', '*').replace('^', '**')))
            if Decimal.is_nan(counting):
                complete = 0
        
        return complete, counting, t

    # Attn-JGNN 推理 (已修复, 见上方 _run_attn_jgnn 方法)


class MISSolver:
    def __init__(self, opts):
        self.opts = opts
        assert self.opts.solver == 'MIS'
        self.exec_dir = os.path.abspath('external/MIS')
        self.cmd_line = ['python', 'mis.py']
    
    def run(self, input_filepath):
        filename = os.path.splitext(os.path.basename(input_filepath))[0]
        tmp_filepath = os.path.join(os.path.dirname(input_filepath), filename + '_' + self.opts.solver + '.out')
        cmd_line = self.cmd_line.copy()
        cmd_line.append(input_filepath)
        cmd_line.append('--out')
        cmd_line.append(tmp_filepath)

        t0 = time.time()
        timeout_expired = 0
        try:
            process = subprocess.Popen(cmd_line, cwd=self.exec_dir, start_new_session=True)
            process.communicate(timeout=self.opts.timeout)
        except:
            timeout_expired = 1
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        t = time.time() - t0

        complete = 0
        ind_vars = None

        if timeout_expired or not os.path.exists(tmp_filepath):
            return complete, ind_vars, t
        
        complete = 1
        with open(tmp_filepath, 'r') as f:
            lines = f.readlines()
            ind_vars = [int(s) for s in lines[0].strip().split()[:-1]]
        
        os.remove(tmp_filepath)
        return complete, ind_vars, t


class MESolver:
    def __init__(self, opts):
        self.opts = opts
        assert self.opts.solver == 'bdd_minisat_all'
        self.exec_dir = os.path.abspath('external/bdd_minisat_all')
        self.cmd_line = ['python', 'bdd_minisat_all.py']
    
    def run(self, input_filepath):
        filename = os.path.splitext(os.path.basename(input_filepath))[0]
        tmp_filepath = os.path.join(os.path.dirname(input_filepath), filename + '_' + self.opts.solver + '.out')
        output_filepath = os.path.join(os.path.dirname(input_filepath), filename + '_' + self.opts.solver + '.pkl')
        cmd_line = self.cmd_line.copy()
        cmd_line.append(input_filepath)
        cmd_line.append(tmp_filepath)
        cmd_line.append(output_filepath)

        t0 = time.time()
        timeout_expired = 0
        try:
            process = subprocess.Popen(cmd_line, cwd=self.exec_dir, start_new_session=True)
            process.communicate(timeout=self.opts.timeout)
        except:
            timeout_expired = 1
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        t = time.time() - t0

        complete = 0
        marginal = None

        if timeout_expired or not os.path.exists(output_filepath):
            if os.path.exists(tmp_filepath):
                os.remove(tmp_filepath)
            return complete, marginal, t
        
        complete = 1
        with open(output_filepath, 'rb') as f:
            marginal = pickle.load(f)
        
        os.remove(tmp_filepath)
        os.remove(output_filepath)

        return complete, marginal, t
