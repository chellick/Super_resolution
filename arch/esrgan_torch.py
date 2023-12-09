import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)



class DenseBlock(nn.Module):
    def __init__(self, filters=64, gc=32):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(filters + 1 * gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(filters + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(filters + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(filters + 4 * gc, gc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
      x1 = self.lrelu(self.conv1(x))
      x2 = self.lrelu(self.conv2(torch.concat((x, x1), 1)))
      x3 = self.lrelu(self.conv3(torch.concat((x, x1, x2), 1)))
      x4 = self.lrelu(self.conv4(torch.concat((x, x1, x2, x3), 1)))
      x5 = self.conv5(torch.concat((x, x1, x2, x3, x4), 1))
      return x5


class RRDB(nn.Module):
    def __init__(self, filters, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = DenseBlock(filters, gc)
        self.RDB2 = DenseBlock(filters, gc)
        self.RDB3 = DenseBlock(filters, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out
    
class ESRGAN(nn.Module):
    def __init__(self, in_nc, out_nc, filters, nb, gc=1):
        super(ESRGAN, self).__init__()
        RRDB_block_f = functools.partial(RRDB, filters=filters, gc=gc)
        self.conv_1 = nn.Conv2d(in_nc, filters, 3, 1, 1)
        self.RRDB_1 = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(filters, filters, 3, 1, 1)
        
        self.upconv_1 = nn.Conv2d(filters, filters, 3, 1, 1)
        self.upconv_2 = nn.Conv2d(filters, filters, 3, 1, 1)
        self.upconv_3 = nn.Conv2d(filters, filters, 3, 1, 1)
        self.upconv_4 = nn.Conv2d(filters, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        fea = self.conv_1(x)
        trunk = self.trunk_conv(self.RRDB_1(fea))
        fea = trunk + fea
        
        fea = self.lrelu(self.upconv_1(F.interpolate(fea, scale_factor=1, mode="nearest")))
        fea = self.lrelu(self.upconv_2(F.interpolate(fea, scale_factor=1, mode="nearest")))
        out = self.upconv_4(self.lrelu(self.upconv_3(fea)))
        return out
    
    
path = 'datasets/unsplash_benchmark/Image Super Resolution - Unsplash/'
hr_path = path + 'high res/'
lr_path = path + 'low res/'

os.listdir(path)

def load_data(path, full_data=True, type='6'):
    if full_data:
        data_path = path
        hr_path = os.path.join(path, 'high res/')
        lr_path = os.path.join(path, 'low res/')
        high_resolution = []
        low_resolution = []

        for i in range(1, 100):
            hr_img = cv2.imread(os.path.join(hr_path, f'{i}.jpg'))
            lr_img = cv2.imread(os.path.join(lr_path, f'{i}_{type}.jpg'))
            if hr_img.shape[0] == 800 and lr_img.shape[0] == 800 and hr_img.shape[1] == 1200 and lr_img.shape[1] == 1200:
              high_resolution.append(hr_img)
              low_resolution.append(lr_img)
            else:
              continue


        high_resolution = np.array(high_resolution)
        low_resolution = np.array(low_resolution)
        # print(high_resolution.shape)
        high_resolution = np.transpose(high_resolution, (0, 3, 1, 2))
        low_resolution = np.transpose(low_resolution, (0, 3, 1, 2))

        high_resolution_tensor = torch.from_numpy(high_resolution).float()
        low_resolution_tensor = torch.from_numpy(low_resolution).float()

        return high_resolution_tensor, low_resolution_tensor
    
start = time.time()
hr, lr = load_data(path)
print(time.time()-start, "Total time")

hr_train, hr_valid, lr_train, lr_valid = train_test_split(hr, lr, test_size=1, random_state=21)

print("Training set shapes - High-res:", hr_train.shape, " Low-res:", lr_train.shape)
print("Validation set shapes - High-res:", hr_valid.shape, " Low-res:", lr_valid.shape)


model = ESRGAN(3, 3, 64, 1, 2)

criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 10

for epoch in range(num_epochs):
    model.train()

    # Iterate over batches in the training set
    for i in range(0, len(hr_train), batch_size):
        hr_batch = hr_train[i:i+batch_size]
        lr_batch = lr_train[i:i+batch_size]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(lr_batch)

        # Compute the loss
        loss = criterion(outputs, hr_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(lr_valid)
        val_loss = criterion(val_outputs, hr_valid)

    # Print training and validation loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Validation Loss: {val_loss.item()}')
