import torch
import torchvision.transforms as transforms

from PIL import Image

def read_img(image_path):
    # 加载图像并转换为 PyTorch 张量
    image_tensor = transforms.ToTensor()(Image.open(image_path).convert('RGB'))
    
    return image_tensor

def gen_grad(image_tensor):
    # 计算梯度，h,w两个维度
    grad_x, grad_y = torch.gradient(image_tensor, dim=[1, 2])
    grad_matrix = torch.sqrt(grad_x**2 + grad_y**2).mean(dim=0)
    
    return grad_matrix

def sample_coords(raw_matrix, l, num):
    h, w = raw_matrix.shape
    grad_sum = torch.sum(raw_matrix)
    prob_matrix = (1-l)*raw_matrix/grad_sum + l/(h*w)

    flat_probs = prob_matrix.view(-1)
    cum_probs = torch.cumsum(flat_probs, dim=0)
    rand_nums = torch.rand(num)
    indices = torch.searchsorted(cum_probs, rand_nums)
    # 将索引转换回二维坐标
    ys = indices // w
    xs = indices % w

    return torch.stack((ys, xs), dim=-1)

def scaled_coords(coords, size):
    u = coords.clone().to(torch.float32)
    u[:, 0] /= size[0]  # h
    u[:, 1] /= size[1]  # w
    
    return u

def compute_l1_loss(tensor1, tensor2):
    tensor1 = tensor1.permute(2, 0, 1) # [3, h, w]
    
    # 确保两个张量的形状相同
    assert tensor1.shape == tensor2.shape, "两个张量的形状必须相同"
    
    # 计算每个像素位置上的绝对差值
    abs_diff = torch.abs(tensor1 - tensor2)
    
    # 将每个像素位置上的三个通道的差值相加
    l1_loss = torch.sum(abs_diff, dim=0)
    
    return l1_loss
