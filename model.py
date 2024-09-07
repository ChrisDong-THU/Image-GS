import torch
import torch.nn as nn

from gaussian import gaussian_2d
from utils import read_img, gen_grad, sample_coords, scaled_coords


class ImageGS(nn.Module):
    def __init__(self, image_path, num, k) -> None:
        super(ImageGS, self).__init__()
        self.k = k # 概率前k个
        
        self.image_tensor = read_img(image_path)
        self.img_size = self.image_tensor.shape[1:]
        self.grad_matrix = gen_grad(self.image_tensor)
        
        self.init_params(l=0.3, num=num)

    def init_params(self, l, num):
        # 中心点坐标，旋转角度，缩放尺度，像素值
        coords = sample_coords(self.grad_matrix, l, num)
        self.u = nn.Parameter(scaled_coords(coords, self.img_size))
        self.t = nn.Parameter(torch.zeros((num)))
        self.s = nn.Parameter(torch.ones((num, 2)) * 2/max(self.img_size))
        self.c = nn.Parameter(self.image_tensor[:,coords[:,0]-1,coords[:,1]-1].transpose(0, 1))
    
    def add_params(self, l, num, loss_matrix):
        new_u = sample_coords(loss_matrix, l, num)
        self.u = nn.Parameter(torch.cat((self.u, scaled_coords(new_u, self.img_size).to('cuda:0')), dim=0))
        new_t = torch.zeros(num).to('cuda:0')
        self.t = nn.Parameter(torch.cat((self.t, new_t), dim=0))
        new_s = (torch.ones((num, 2)) * 2/max(self.img_size)).to('cuda:0')
        self.s = nn.Parameter(torch.cat((self.s, new_s), dim=0))
        new_c = self.image_tensor[:, new_u[:, 0]-1, new_u[:, 1]-1].transpose(0, 1).to('cuda:0')
        self.c = nn.Parameter(torch.cat((self.c, new_c), dim=0))
        
    def cal_colors(self, coords):
        probs = gaussian_2d(coords, self.u, self.t, self.s)
        topk_probs, topk_indices = torch.topk(probs, self.k, dim=1)
        
        # 直接索引颜色并加权平均
        topk_colors = self.c[topk_indices] # [num, num_dis, R,G,B]
        colors = torch.sum(topk_colors * topk_probs.unsqueeze(-1), dim=1) / (torch.sum(topk_probs, dim=1, keepdim=True) + 1e-8) # [num, RGB]
        
        return colors
    
    def forward(self, x):
        # [batch_size, num, coord]
        x_color = self.cal_colors(scaled_coords(x.view(-1, *x.shape[2:]), self.img_size)) # [batch_size*num, coord]
        x_color = x_color.view(*x.shape[:2], *x_color.shape[1:]) # [batch_size, num, color]

        return x_color


if __name__=='__main__':
    img_gs = ImageGS('tree.jpg', 100, 5)
    x = torch.tensor([[[10,10], [50,50],[90,90]],[[20,20], [60,60],[80,80]]])

    y = img_gs(x)
    pass