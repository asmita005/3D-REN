import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class RegionEnsemble(nn.Module):

    def __init__(self, feat_size=11):
        #assert((feat_size/4).is_integer())
        super(RegionEnsemble, self).__init__()
        self.feat_size = int((feat_size-1)/2)
        self.grids = nn.ModuleList()
        
        self.grids.append(nn.Sequential(nn.Linear(64*self.feat_size*self.feat_size*self.feat_size, 2048), nn.ReLU(), nn.Dropout(), nn.Linear(2048,2048), nn.ReLU(), nn.Dropout()))

        self.grids.append(nn.Sequential(nn.Linear(64*self.feat_size*self.feat_size*(self.feat_size+1), 2048), nn.ReLU(), nn.Dropout(), nn.Linear(2048,2048), nn.ReLU(), nn.Dropout()))
        self.grids.append(nn.Sequential(nn.Linear(64*self.feat_size*(self.feat_size+1)*self.feat_size, 2048), nn.ReLU(), nn.Dropout(), nn.Linear(2048,2048), nn.ReLU(), nn.Dropout()))
        self.grids.append(nn.Sequential(nn.Linear(64*self.feat_size*(self.feat_size+1)*(self.feat_size+1), 2048), nn.ReLU(), nn.Dropout(), nn.Linear(2048,2048), nn.ReLU(), nn.Dropout()))

        self.grids.append(nn.Sequential(nn.Linear(64*(self.feat_size+1)*self.feat_size*self.feat_size, 2048), nn.ReLU(), nn.Dropout(), nn.Linear(2048,2048), nn.ReLU(), nn.Dropout()))
        self.grids.append(nn.Sequential(nn.Linear(64*(self.feat_size+1)*self.feat_size*(self.feat_size+1), 2048), nn.ReLU(), nn.Dropout(), nn.Linear(2048,2048), nn.ReLU(), nn.Dropout()))
        self.grids.append(nn.Sequential(nn.Linear(64*(self.feat_size+1)*(self.feat_size+1)*self.feat_size, 2048), nn.ReLU(), nn.Dropout(), nn.Linear(2048,2048), nn.ReLU(), nn.Dropout()))

        self.grids.append(nn.Sequential(nn.Linear(64*(self.feat_size+1)*(self.feat_size+1)*(self.feat_size+1), 2048), nn.ReLU(), nn.Dropout(), nn.Linear(2048,2048), nn.ReLU(), nn.Dropout()))		

    def forward(self, x):

        m = self.feat_size
        regions = []
        ensemble = []

        # 8 corners
        regions += [x[:, :, :m, :m, :m].clone(), x[:, :, :m, :m, m:].clone(), x[:, :, :m, m:, :m].clone(), x[:, :, :m, m:, m:].clone(), x[:, :, m:, :m, :m].clone(), x[:, :, m:, :m, m:].clone(),x[:, :, m:, m:, :m].clone(), x[:, :, m:, m:, m:].clone()]
        #regions += [x[:, :, :m, :m, :m].clone(), x[:, :, :m, :m, (m-1):].clone(), x[:, :, :m, (m-1):, :m].clone(), x[:, :, :m, (m-1):, (m-1):].clone(), x[:, :, (m-1):, :m, :m].clone(), x[:, :, (m-1):, :m, (m-1):].clone(),x[:, :, (m-1):, (m-1):, :m].clone(), x[:, :, (m-1):, (m-1):, (m-1):].clone()]

        
        for i in range(0,8):
            out = regions[i]
            # print(out.shape)
            out = out.contiguous()
            out = out.view(out.size(0),-1)
            out = self.grids[i](out)
            ensemble.append(out)

        out = torch.cat(ensemble,1)

        return out

class Basic3DBlock(nn.Module):
     def __init__(self, in_planes, out_planes,kernel_size):
         super(Basic3DBlock, self).__init__()
         self.block = nn.Sequential(
             nn.Conv3d(in_planes, out_planes,kernel_size=kernel_size, stride=1, padding=((kernel_size -1) // 2)),
             nn.BatchNorm3d(out_planes),
             nn.ReLU(True)
         )

     def forward(self, x):
         return self.block(x)


class ConvBlock(nn.Module):
     def __init__(self, in_planes, out_planes, kernel_size):
         super(ConvBlock, self).__init__()
         self.conv = nn.Sequential(
             nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size -1) // 2))
         )

     def forward(self, x):
         return self.conv(x)


class Res3DBlock(nn.Module):
     def __init__(self, in_planes, out_planes):
         super(Res3DBlock, self).__init__()
         self.res_branch = nn.Sequential(
             nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm3d(out_planes),
             nn.ReLU(True),
             nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm3d(out_planes)
         )

         if in_planes == out_planes:
             self.skip_con = nn.Sequential()
         else:
             self.skip_con = nn.Sequential(
                 nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                 nn.BatchNorm3d(out_planes)
             )

     def forward(self, x):
         res = self.res_branch(x)
         skip = self.skip_con(x)
         return F.relu(res + skip, True)


class Pool3DBlock(nn.Module):
     def __init__(self, pool_size):
         super(Pool3DBlock, self).__init__()
         self.pool_size = pool_size

     def forward(self, x):
         return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)


class V2VModel(nn.Module):
     def __init__(self, input_channels, output_channels):
         super(V2VModel, self).__init__()
         
         feat = np.floor(((88 - 1 -1)/2) +1)
         feat = np.floor(((feat - 1-1)/2) +1)
         feat = np.floor(((feat - 1-1)/2) +1)

         self.front_layers = nn.Sequential(
             Basic3DBlock(input_channels, 16, 3),
             ConvBlock(16,16,3), #check
             Pool3DBlock(2),
             ConvBlock(16, 32, 1), # check
             nn.ReLU(True),
             Res3DBlock(32, 32),
             Pool3DBlock(2), #check
             nn.ReLU(True), #repeat
             ConvBlock(32, 64, 1),  # check
             nn.ReLU(True),
             Res3DBlock(64, 64),
             Pool3DBlock(2),  # check
             nn.ReLU(True),
             RegionEnsemble(feat_size=feat),
             nn.Linear(8*2048, output_channels*3)

             #TODO now slicing

         )
         self._initialize_weights()

     def forward(self, x):
         x = self.front_layers(x)
         return x
	
     def _initialize_weights(self):
         for m in self.modules():
             if isinstance(m, nn.Conv3d):
                 nn.init.normal_(m.weight, 0, 0.001)
                 nn.init.constant_(m.bias, 0)
             elif isinstance(m, nn.ConvTranspose3d):
                 nn.init.normal_(m.weight, 0, 0.001)
                 nn.init.constant_(m.bias, 0)
