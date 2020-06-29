import cv2
import torch
import skimage.io
import numpy as np
from torch.utils import data


class ImageDataSet(data.Dataset):

    def __init__(self, tile_records, data_dir, mask_dir, batch_size=8, num_chan=3, tile_size=224, num_class=6, smoothing=0.02, shuffle=True):
        self.tile_records = tile_records
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.num_chan = num_chan
        self.tile_size = tile_size
        self.num_class = num_class
        self.smoothing = smoothing
        self.shuffle = shuffle
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:' + str(torch.cuda.current_device())

    # Denotes the total number of samples
    def __len__(self):
        return len(self.tile_records)

    # Generate one sample of data
    def __getitem__(self, index):
        tile_info = self.tile_records[index].split(',')
        filename = tile_info[0]
        kdx = int(tile_info[1])
        jdx = int(tile_info[2])
        label = int(tile_info[3])
        biopsy = skimage.io.MultiImage(self.data_dir + '\\' + filename + '.tiff')
        img = biopsy[-1]

        # pad image so that all tiles are full height/width
        hr = self.tile_size - img.shape[0] % self.tile_size
        wr = self.tile_size - img.shape[1] % self.tile_size
        img = cv2.copyMakeBorder(img, 0, hr, 0, wr, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Initialize arrays, scale with mean and stddev, transpose to channels first
        img_tile = img[kdx: kdx + self.tile_size, jdx: jdx + self.tile_size, :]
        mean = np.mean(img_tile)
        stddev = np.std(img_tile)
        img_tile = (img_tile - mean) / (stddev + 1.0e-8)
        img_tile = np.swapaxes(img_tile, 0, 2) 
        img_tile = np.swapaxes(img_tile, 1, 2)

        X = torch.from_numpy(img_tile).to(dtype=torch.float32, device=self.device)
        # One Hot Encoding
        # y = torch.zeros((self.num_class), dtype=torch.float32, device=self.device)
        # y[label] = 1.0
        # Use Label Smoothing
        y = torch.tensor([self.smoothing for _ in range(self.num_class)], dtype=torch.float32, device=self.device)
        y[label] = 1.0 - (self.num_class - 1) * self.smoothing
        return X, y

