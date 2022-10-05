from .feat_basetransformer3_2d import FEATBaseTransformer3_2d
from .feat_basetransformer2 import get_k_base, ids2classes
import torch
import numpy as np
import torch.nn.functional as F
import os
import os.path as osp





def load_numpy(name):
    return np.load(os.path.join('/home/mayug/projects/few_shot/notebooks/temp/', name+'.npy'))

class FEATBaseTransformer3_2d_patch(FEATBaseTransformer3_2d):
    def __init__(self, args):
        if args.backbone_class == 'ConvNet':
            args.max_pool = 'max_pool'
        else:
            args.max_pool = 'avg_pool'
        args.resize = True
        super().__init__(args)
        # these 2d embeddings of base instances are used for combination
        if args.dataset == 'MiniImageNet':
            if args.backbone_class == 'ConvNet':
                print('Loading embeds_cache_cnn4_2d_patches.pt')
                proto_dict_2d = torch.load(
                    osp.join(self.embeds_cache_root, 'embeds_cache_cnn4_2d_patches.pt'))                
            elif args.backbone_class == 'Res12':
                raise NotImplementedError
                print('Loading embeds_cache_res12_2d_patch_just_init.pt')
                proto_dict_2d = torch.load(
                    osp.join(self.embeds_cache_root, 'embeds_cache_res12_2d_just_init.pt'))

        elif args.dataset == 'CUB':
            raise NotImplementedError

        self.proto_dict_2d = proto_dict_2d
        self.all_proto_2d = proto_dict_2d['embeds'].cuda()
        self.n_patches = args.n_patches

    # Implementing patcher function
    def patch(self, x):
        n_patches = self.n_patches
        total_patches = n_patches*n_patches
        patch_size = int(x.shape[-1]/n_patches)
        n_channels = x.shape[1]
        n_instances = x.shape[0]
        hdim = self.hdim

        a = torch.cat(
            [x[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size].unsqueeze(1) 
            for i in range(n_patches) for j in range(n_patches)], dim=1)
        a = a.reshape(-1, n_channels, patch_size, patch_size)
        # might have to batch it here to make it fit
        a = self.encoder(a)
        # print('check a here ', a.shape)
        a = a.reshape(n_instances, total_patches, hdim)
        # print('check a here ', a.shape)
        a = a.permute(0, 2, 1).contiguous()
        # print('check a here ', a.shape)
        # print('hdim ', hdim)
        a = a.reshape(n_instances, hdim, n_patches, n_patches)

        # print('a.shape')
        return a
    # Implementing forward again, to get patchwise embeddings

    def forward(self, x, ids=None,  get_feature=False):

            
        # print('inside base', [x.shape, x.min(), x.max()])
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            # print('here x shape ', x.shape)
            instance_embs = self.patch(x)
            # print('after patching  instance_embs shape ', instance_embs.shape)
            num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)
            # print('support and query idx')
            # print(support_idx.shape)s
            # asd
            if self.training:
                if self.args.pass_ids:
                    logits, logits_reg = self._forward(instance_embs, support_idx, query_idx, ids)
                else:
                    logits, logits_reg = self._forward(instance_embs, support_idx, query_idx)
                return logits, logits_reg
            else:
                if self.args.pass_ids:
                    logits = self._forward(instance_embs, support_idx, query_idx, ids)
                else:
                    logits = self._forward(instance_embs, support_idx, query_idx)
                return logits


    def _forward(self, instance_embs, support_idx, query_idx, ids=None):
        # print('checking 1', [instance_embs.shape,instance_embs.min(),instance_embs.max()])
        spatial_dim = instance_embs.shape[-1]
        self.spatial_dim = spatial_dim
        self.save_as_numpy(instance_embs, 'instance_embs')

        emb_dim = instance_embs.size(-3)
        num_patches = np.prod(instance_embs.shape[-2:]) 

        # print("emb_dim, num_patches", [emb_dim, num_patches])
        # print('instance_embs', instance_embs.shape)
        # print('support_idx, query_idx', [support_idx.shape, query_idx.shape])
        # organize support/query data

        # print(instance_embs.shape)
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (emb_dim, spatial_dim, spatial_dim,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (emb_dim, spatial_dim, spatial_dim,)))
    
        # print('support, query', [support.shape, query.shape])
        self.save_as_numpy(support, 'support')
        self.save_as_numpy(support, 'query')
        # support n_class * K * (5 * 5) * 64
        # get mean of the support
        proto = support.mean(dim=1) 
        self.save_as_numpy(proto, 'proto')
        # proto 1 * n_class * (5 * 5) * 64
        n_class = proto.shape[1]
        n_batch = proto.shape[0]
        k = self.args.k - 1 
        # print('proto after mean ', proto.shape)

        # get topk 3d base protos is using the same 1d topk function
        
        # with torch.no_grad():
        base_protos = self.get_base_protos(proto, ids) # 
        self.save_as_numpy(base_protos, 'base_protos')
        # base_protos n_class * topk * 64 * (5 * 5)

        # print('base_protos', base_protos.shape)
        # print('proto ', proto.shape)
        # proto = proto.reshape(proto.shape[1], -1, emb_dim)
        proto = proto.reshape(proto.shape[1], emb_dim, -1).permute(0, 2, 1).contiguous()
        self.save_as_numpy(proto, 'proto_after_reshape')
        # print('proto reshaped', proto.shape)
        # proto n_class * (5*5) * 64

        # print('proto.shape ', proto.shape)

        # combined_protos = base_protos.reshape(n_class*n_batch, k*num_patches, emb_dim)

        base_protos = base_protos.permute(0, 2, 1, 3, 4).contiguous()
        combined_protos = base_protos.reshape(n_class*n_batch, emb_dim, -1).permute(0, 2, 1).contiguous()
        self.save_as_numpy(combined_protos, 'combined_protos')

        # print('combined_protos.shape ', combined_protos.shape)
        # print('proto.shape ', proto.shape)
        # print('combined_protos.shape ', combined_protos.shape)

    
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        print('before slf attention proto', [self.training, proto.shape, proto.min(),proto.max()])

        print('before slf attention  combined proto', [combined_protos.shape,combined_protos.min(),combined_protos.max()])
        
        if self.feat_attn==2:
            proto = self.self_attn2(proto, proto, proto)
        proto = self.slf_attn(proto, combined_protos, combined_protos)
        print('after slf attention proto', [self.training, proto.shape, proto.min(),proto.max()])

        # print('after slf attention proto', [proto.shape])
        proto = proto.permute(0, 2, 1).contiguous()
        # print('proto after permute nad before making 2d again ', proto.shape)
        
        proto = proto.view(-1, emb_dim, spatial_dim, spatial_dim)
        # print('proto after making 2d and before maxpool ', proto.shape)

        # mean for proto; unhide below
        # proto = proto.mean(dim=1).unsqueeze(0)
  
        # print('after maxpool proto', [proto.shape])
        # print('query before view operation', query.shape)
        query = query.view(-1, emb_dim, spatial_dim, spatial_dim)
        # print('query before maxpool shape', [query.shape])
        # print('query after maxpool shape', [query.shape])

        if isinstance(self.embed_pool, torch.nn.Identity):
            # print(proto.shape)
            # print(query.shape)
            proto = proto.reshape(proto.shape[0], -1).unsqueeze(0)
            query = query.reshape(query.shape[0], -1)

            emb_dim = emb_dim*(spatial_dim**2)

            # asd
        else:
            proto = self.embed_pool(proto, kernel_size=5).squeeze().unsqueeze(0)
            query = self.embed_pool(query, kernel_size=5).squeeze()
       
            
  

        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])
        # print('after slf attention proto', [proto.shape])

        if self.feat_attn==1:
            proto = self.self_attn2(proto, proto, proto)

        self.after_attn = proto

        # print('after attention ', proto.shape)
        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
    
            logits = - torch.mean((proto - query) ** 2, 2) / self.args.temperature
            print('logits ', [logits.shape, logits.min(), logits.max()])
            asd
        else:
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0,2,1]).contiguous()) / self.args.temperature
            logits = logits.view(-1, num_proto)
        
        # for regularization
        if self.training:
            # return logits, None
            # TODO this can be further adapted for basetransformer version
            if self.args.balance==0:
                return logits, None
            aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
                                  query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
            # apply the transformation over the Aug Task
            if self.feat_attn==1:
                aux_emb = self.self_attn2(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
            else:
                # print('before aux forward pass ', [aux_task.shape, combined_protos.shape])
                aux_emb = self.slf_attn(aux_task, combined_protos, combined_protos)
                # print('after aux forward pass ', aux_emb.shape)
            # compute class mean
            aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
            aux_center = torch.mean(aux_emb, 2) # T x N x d
            
            if self.args.use_euclidean:
                aux_task = aux_task.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
                aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
                aux_center = aux_center.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
                # print('aux center ', aux_center.shape)
                # print('aux_task ', aux_task.shape)
                logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
            else:
                aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
                aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
    
                logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1]).contiguous) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)            
            
            return logits, logits_reg            
        else:
            return logits 