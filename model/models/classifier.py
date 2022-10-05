import torch
import torch.nn as nn
import numpy as np
from model.utils import euclidean_metric
import torch.nn.functional as F
    
class Classifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            hdim = 64
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)                        
        else:
            raise ValueError('')

        self.fc = nn.Linear(hdim, args.num_class)
        self.n_patches = args.n_patches
        self.hdim = hdim

    def forward(self, data, is_emb = False):
        if self.n_patches is not None:
            # print('data ', data.shape)
            data = self.patch(data)
            out = F.max_pool2d(data, kernel_size=self.n_patches).squeeze()
            # print('out ', out.shape)
            # asd
        else:
            out = self.encoder(data)
            # print('out ', out.shape)
            if not is_emb:
                out = self.fc(out)
                # print('out fter fc ', out.shape)
        return out
   
    # Implementing patcher function
    def patch(self, x):
        n_patches = self.n_patches
        total_patches = n_patches*n_patches
        patch_size = int(x.shape[-1]/n_patches)
        n_channels = x.shape[1]
        n_instances = x.shape[0]
        hdim = self.hdim
        # print('before patching x ', x.shape)
        a = torch.cat(
            [x[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size].unsqueeze(1) 
            for i in range(n_patches) for j in range(n_patches)], dim=1)
        a = a.reshape(-1, n_channels, patch_size, patch_size)
        # might have to batch it here to make it fit
        # print('a ', a.shape)
        a = self.encoder(a)
        # print('check a here ', a.shape)
        a = a.reshape(n_instances, total_patches, hdim)
        # print('check a here ', a.shape)
        a = a.permute(0, 2 , 1).contiguous()
        # print('check a here ', a.shape)
        # print('hdim ', hdim)
        a = a.reshape(n_instances, hdim, n_patches, n_patches)

        # print('a.shape final ', a.shape)
        return a    

    def forward_proto(self, data_shot, data_query, way = None):
        if way is None:
            way = self.args.num_class
        # print('proto before encoder ', data_shot.shape)
        # print('data query mean 5 ', data_query[5, :, :, :].mean())
        # print('data shot, data query ', [data_shot.shape, data_query.shape])
        if self.n_patches is not None:
            proto = self.patch(data_shot)
            proto = F.max_pool2d(proto, kernel_size=self.n_patches).squeeze()
            query = self.patch(data_query)
            query = F.max_pool2d(query, kernel_size=self.n_patches).squeeze()
        else:
            proto = self.encoder(data_shot)
            query = self.encoder(data_query)
        # print('proto after encoder', proto.shape)
        # print(proto.mean(1))
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)
        # print('proto after reshape and mean', proto.shape)
        # print(proto.mean(1))
        
        # print('embed query mean 5 ', query.mean(1))
        # print(query.shape)
        logits_dist = euclidean_metric(query, proto)
        logits_sim = torch.mm(query, F.normalize(proto, p=2, dim=-1).t())
        return logits_dist, logits_sim