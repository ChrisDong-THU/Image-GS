import torch
        
import torch

# def gaussian_2d(x, mus, thetas, s):
#     '''
#     计算多个二维高斯分布在点x处的概率

#     :param tensor x: 二维坐标向量 (num_points, 2)
#     :param tensor mus: 二维均值向量 (num_distributions, 2)
#     :param tensor thetas: 旋转角度 (num_distributions,)
#     :param tensor s: 缩放因子 (num_distributions, 2)
#     :return tensor: 概率密度值 (num_points, num_distributions)
#     '''
#     # 获取设备
#     device = x.device

#     # 检查输入维度
#     assert x.shape[-1] == 2, "x should have shape (num_points, 2)"
#     assert mus.shape[-1] == 2, "mus should have shape (num_distributions, 2)"
#     assert s.shape[-1] == 2, "s should have shape (num_distributions, 2)"

#     cos_rot = torch.cos(thetas)
#     sin_rot = torch.sin(thetas)

#     R = torch.stack([
#         torch.stack([cos_rot, -sin_rot], dim=-1),
#         torch.stack([sin_rot, cos_rot], dim=-1)
#     ], dim=-2).to(torch.float32)

#     # 构造缩放矩阵
#     S = torch.diag_embed(s).to(torch.float32).to(device)

#     # 计算 RSS^TR^T
#     covars = R @ S @ S @ R.transpose(-1,-2)
    
#     # 计算逆协方差矩阵，添加正则化项
#     inv_covars = torch.inverse(covars + 1e-8 * torch.eye(covars.size(-1)).to(covars.device))  # shape: (num_distributions, 2, 2)

#     # 计算差值
#     diff = x.unsqueeze(1) - mus.unsqueeze(0)  # shape: (num_points, num_distributions, 2)
#     # 计算指数部分
#     exp_part = -0.5 * torch.einsum('ijk,jkl,ijl->ij', diff, inv_covars, diff)

#     # 计算概率密度
#     probs = torch.exp(exp_part)

#     return probs


def gaussian_2d(x, mus, thetas, s, batch_size=1024):
    '''
    计算多个二维高斯分布在点x处的概率，进行分块处理以减少显存占用

    :param tensor x: 二维坐标向量 (num_points, 2)
    :param tensor mus: 二维均值向量 (num_distributions, 2)
    :param tensor thetas: 旋转角度 (num_distributions,)
    :param tensor s: 缩放因子 (num_distributions, 2)
    :param int batch_size: 分块计算的大小，默认 10000
    :return tensor: 概率密度值 (num_points, num_distributions)
    '''
    # 获取设备
    device = x.device

    # 计算旋转矩阵
    cos_rot = torch.cos(thetas)
    sin_rot = torch.sin(thetas)

    R = torch.stack([
        torch.stack([cos_rot, -sin_rot], dim=-1),
        torch.stack([sin_rot, cos_rot], dim=-1)
    ], dim=-2)

    # 构造缩放矩阵
    S = torch.diag_embed(s).to(device)

    # 计算协方差矩阵
    covars = R @ S @ S @ R.transpose(-1,-2)

    # 计算逆协方差矩阵，添加正则化项
    inv_covars = torch.inverse(covars + 1e-8 * torch.eye(covars.size(-1)).to(covars.device))

    # 准备结果存储，初始化一个空的概率密度张量
    num_points = x.size(0)
    num_distributions = mus.size(0)
    probs = torch.zeros((num_points, num_distributions), device=device)

    # 分块处理 x (逐批计算 diff 和 exp_part)
    for i in range(0, num_points, batch_size):
        # 当前批次的终点，确保最后一个批次的处理
        end_idx = min(i + batch_size, num_points)
        
        # 当前批次的坐标点
        x_batch = x[i:end_idx]  # 确保取出不超过范围的批次
        
        # 计算当前批次的差值 diff
        diff = x_batch.unsqueeze(1) - mus.unsqueeze(0)  # shape: (batch_size, num_distributions, 2)
        
        # 计算指数部分，批量处理
        exp_part_batch = -0.5 * torch.einsum('ijk,jkl,ijl->ij', diff, inv_covars, diff)

        # 计算概率密度并存储
        probs[i:end_idx] = torch.exp(exp_part_batch)
        torch.cuda.empty_cache()

    return probs