import torch
import math

def rmse(pred_logz, true_logz):
    """
    计算对数模型数的均方根误差（RMSE）
    参数:
        pred_logz: Tensor, 预测的对数模型数
        true_logz: Tensor, 真实的对数模型数
    返回:
        rmse: float
    """
    mse = torch.mean((pred_logz - true_logz) ** 2)
    return torch.sqrt(mse).item()

def relative_error(pred_count, true_count):
    """
    计算相对误差 |pred - true| / max(pred, true)
    参数:
        pred_count: float or Tensor, 预测的模型数（线性尺度）
        true_count: float or Tensor, 真实的模型数
    返回:
        rel_err: float
    """
    pred = pred_count if isinstance(pred_count, float) else pred_count.item()
    true = true_count if isinstance(true_count, float) else true_count.item()
    denom = max(pred, true)
    if denom == 0:
        return 0.0
    return abs(pred - true) / denom

def within_tolerance(pred_logz, true_logz, epsilon=0.5):
    """
    判断预测值是否在真值 ± epsilon（对数尺度）范围内
    返回: bool
    """
    return abs(pred_logz - true_logz) <= epsilon

def compute_all_metrics(pred_logz_list, true_logz_list):
    """
    批量计算多个实例的 RMSE、平均相对误差、达标率
    参数:
        pred_logz_list: List[Tensor] 或 List[float]
        true_logz_list: List[Tensor] 或 List[float]
    返回:
        metrics: dict
    """
    pred = torch.tensor(pred_logz_list)
    true = torch.tensor(true_logz_list)
    rmse_val = torch.sqrt(torch.mean((pred - true) ** 2)).item()
    abs_err = torch.abs(pred - true)
    within_05 = (abs_err <= 0.5).float().mean().item()
    within_10 = (abs_err <= 1.0).float().mean().item()
    # 相对误差（线性尺度）
    pred_lin = torch.exp(pred)
    true_lin = torch.exp(true)
    rel_err = torch.abs(pred_lin - true_lin) / torch.max(pred_lin, true_lin)
    mean_rel_err = rel_err.mean().item()
    return {
        'RMSE': rmse_val,
        'within_0.5': within_05,
        'within_1.0': within_10,
        'mean_relative_error': mean_rel_err
    }