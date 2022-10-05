import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('/home/mayug/projects/few_shot/notebooks/')
# from sal_utils import get_saliency_crops2, get_sal
from torch.cuda.amp import autocast

class FewShotModel(nn.Module):
    def __init__(self, args, resize=True, sal=False, max_pool='max_pool'):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet(resize=resize, sal=sal, max_pool=max_pool)
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = args.dim_model
            from model.networks.res12 import ResNet
            self.encoder = ResNet(avg_pool=args.max_pool, resize=args.resize, 
                drop_rate=args.drop_rate, out_dim=hdim)
        elif args.backbone_class == 'Res12_info':
            hdim = args.dim_model
            from model.networks.res12_info import ResNet
            self.encoder = ResNet(avg_pool=args.max_pool, resize=args.resize, 
                drop_rate=args.drop_rate, out_dim=hdim)
        elif args.backbone_class == 'WRN_S2M2':
            hdim = args.dim_model
            from model.networks.wrnet_s2m2 import WideResNet28_10
            self.encoder = WideResNet28_10(avg_pool=args.max_pool, resize=args.resize)
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet(pool=args.max_pool, resize=args.resize)
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            print(['resize, max_pool', resize, max_pool])
            self.encoder = Wide_ResNet(28, 10, 0.5, resize=resize, avg_pool=max_pool)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        elif args.backbone_class == 'Res12_ptcv':
            hdim = 512
            from pytorchcv.model_provider import get_model as ptcv_get_model
            resnet_new = ptcv_get_model('resnet12', pretrained=True)
            self.encoder = list(resnet_new.children())[0]
            if args.model_class == 'FEATBaseTransformer3_2d':
                self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-1])
        else:
            raise ValueError('')
        self.sal_crop = args.sal_crop
        self.hdim = hdim
    def split_instances(self, data):
        args = self.args
        if self.training:
            return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot)).long().view(1, args.eval_shot, args.eval_way), 
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way))

    
    def forward(self, x, ids=None,  get_feature=False, simclr_images=None):

        # with autocast():
            # input sal crop
        # if self.sal_crop is not None:
        #     if self.sal_crop == 'saliency_crop2':
        #         x = get_saliency_crops2(x,
        #                 use_minmax=False)
        #     elif self.sal_crop == 'saliency_crop4':
        #         sal = get_sal(x)
        #         sal = sal.repeat_interleave(3, dim=1)
        #         sal = sal*5.4-2.7
        #         # print('sal shape', [sal.shape, sal.min(), sal.max()])
        #         self.sal_embs= self.encoder(sal)
            
        # print('inside base', [x.shape, x.min(), x.max()])
        if get_feature:
            # get feature with the provided embeddings
            print('inside get_featur/e')
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)
            num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)
            # print('support and query idx')
            # print(support_idx)
            # print(query_idx)
            simclr_embs=None
            if simclr_images is not None:
                n_embs, n_views, n_ch, spatial, _ = simclr_images.shape
                simclr_images = simclr_images.reshape(-1, n_ch, spatial, spatial)
                simclr_embs = self.encoder(simclr_images)
                spatial_out = simclr_embs.shape[-1]
                # print('simclr embs ', [simclr_embs.shape, simclr_embs.min(), simclr_embs.max()])

                simclr_embs = simclr_embs.reshape(n_embs, n_views, self.hdim, spatial_out, spatial_out)
                # print('simclr embs ', [simclr_embs.shape, simclr_embs.min(), simclr_embs.max()])
                # print('instance embs ', [instance_embs.shape, instance_embs.min(), instance_embs.max()])

            if self.training:
                if self.args.pass_ids:
                    logits, logits_reg = self._forward(instance_embs, 
                        support_idx, query_idx, ids, simclr_embs=simclr_embs)

                else:
                    logits, logits_reg = self._forward(instance_embs, 
                        support_idx, query_idx, simclr_embs=simclr_embs)
                return logits, logits_reg
            else:
                if self.args.pass_ids:
                    logits = self._forward(instance_embs, support_idx, query_idx, ids)
                else:
                    logits = self._forward(instance_embs, support_idx, query_idx)
                return logits

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')