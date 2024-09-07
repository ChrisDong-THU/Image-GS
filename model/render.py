import torch
import torch.nn.functional as F
from PIL import Image

from .utils import scaled_coords


def gaussian_2d_splatting(h, w, model):
    # 获取设备
    device = next(model.parameters()).device
    
    # 生成并标量化行和列坐标，直接在需要的设备上生成
    rows = torch.linspace(0, 1, h, device=device)
    cols = torch.linspace(0, 1, w, device=device)
    
    # 使用 meshgrid 生成所有可能的坐标对
    row_coords, col_coords = torch.meshgrid(rows, cols, indexing='ij')
    coords = torch.stack((row_coords.flatten(), col_coords.flatten()), dim=1)
    
    # 直接在设备上计算颜色
    with torch.no_grad():
        colors = model.cal_colors(coords)

    # 生成图像张量
    image_tensor = colors.view(h, w, 3)
    
    return image_tensor

def save_img(image_tensor, epoch, loss):
    image_np = (image_tensor.cpu() * 255).clamp(0, 255).to(torch.uint8).numpy()
    image = Image.fromarray(image_np)
    image.save(f'./tmp/epoch_{epoch}_loss_{loss:.5f}.jpg')
