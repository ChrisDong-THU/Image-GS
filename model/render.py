import torch
import torch.nn.functional as F
from PIL import Image

from .utils import scaled_coords

def gaussian_2d_splatting(h, w, model):
    # 获取设备并切换到 CPU
    pre_device = next(model.parameters()).device
    model.to('cpu')
    
    # 生成所有可能的行坐标和列坐标
    rows = torch.arange(h)
    cols = torch.arange(w)
    
    # 使用 meshgrid 生成所有可能的坐标对
    row_coords, col_coords = torch.meshgrid(rows, cols, indexing='ij')
    
    # 将坐标组合成形状为[num, 2]的张量
    coords = torch.stack((row_coords.flatten(), col_coords.flatten()), dim=1)
    coords = scaled_coords(coords, (h, w))
    colors = model.cal_colors(coords)
    
    # 切换回原始设备
    model.to(pre_device)

    # 生成图像张量
    image_tensor = colors.view(h, w, 3)
    
    return image_tensor

def save_img(image_tensor, epoch, loss):
    image_np = (image_tensor * 255).clamp(0, 255).to(torch.uint8).numpy()
    image = Image.fromarray(image_np)
    image.save(f'./tmp/epoch_{epoch}_loss_{loss:.5f}.jpg')
