import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model.model import ImageGS
from model.utils import compute_l1_loss, read_img, sample_coords, gen_grad
from model.render import gaussian_2d_splatting, save_img

filename = './data/ewm_1000.jpg'

image_tensor = read_img(filename)
grad_matrix = gen_grad(image_tensor)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ImageGS(filename, 1000, 10).to(device)
model.train()

lr = np.array([2e-4, 2e-3, 1e-3, 1e-3])

# 创建优化器
optimizer = optim.Adam([
    {'params': model.u, 'lr': lr[0]},  # u
    {'params': model.t, 'lr': lr[1]},   # t
    {'params': model.s, 'lr': lr[2]},  # s
    {'params': model.c, 'lr': lr[3]}    # c
])

# 定义损失函数
loss_function = nn.L1Loss()

for epoch in range(2001):
    coords = sample_coords(grad_matrix, 0.8, 10000) if epoch<=500 else sample_coords(loss_matrix, 0.5, 10000)
    colors = image_tensor[:, coords[:, 0]-1, coords[:, 1]-1].transpose(0, 1)
    
    coords, colors = coords.unsqueeze(0).to(device), colors.unsqueeze(0).to(device)
    pre = model(coords)
    loss = loss_function(colors, pre)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss}')
    
    if epoch%250 == 0 and epoch >= 1:
        recon_img = gaussian_2d_splatting(*model.img_size, model) # 支持超分辨率渲染
        save_img(recon_img, epoch, loss)
        if epoch<1000 :
            loss_matrix = compute_l1_loss(recon_img, model.image_tensor)
            model.add_params(0, 250, loss_matrix.to('cpu'))
        else:
            lr = lr*0.1
        optimizer = optim.Adam([
            {'params': model.u, 'lr': lr[0]},  # u
            {'params': model.t, 'lr': lr[1]},   # t
            {'params': model.s, 'lr': lr[2]},  # s
            {'params': model.c, 'lr': lr[3]}    # c
        ])
