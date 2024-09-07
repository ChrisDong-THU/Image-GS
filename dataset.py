import torch

from torch.utils.data import Dataset
from utils import read_img, sample_coords, gen_grad

class ImageData(Dataset):
    def __init__(self, image_path, data_num, coord_num=2500) -> None:
        super().__init__()
        self.data_num = data_num
        self.coord_num = coord_num
        
        self.image_tensor = read_img(image_path)
        self.grad_matrix = gen_grad(self.image_tensor)
        
        coords_list = [sample_coords(self.grad_matrix, 0.8, self.coord_num) for _ in range(data_num)]
        labels_list = [self.image_tensor[:, coords[:, 0], coords[:, 1]].transpose(0, 1) for coords in coords_list]

        self.coords = torch.stack(coords_list)
        self.labels = torch.stack(labels_list)
                
    def __getitem__(self, index):

        return self.coords[index,:], self.labels[index,:]
    
    def __len__(self):
        
        return self.data_num