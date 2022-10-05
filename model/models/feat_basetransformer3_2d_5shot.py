from .feat_basetransformer3 import FEATBaseTransformer3
from .feat_basetransformer2 import get_k_base, ids2classes
import torch
import numpy as np
import torch.nn.functional as F
import os


    
def load_numpy(name):
    return np.load(os.path.join('/home/mayug/projects/few_shot/notebooks/temp/', name+'.npy'))

class FEATBaseTransformer3_2d_5shot(FEATBaseTransformer3):
    def __init__(self, args):
        args.max_pool = None
        args.resize = False
        super().__init__(args)
        # these 2d embeddings of base instances are used for combination
        if args.embeds_cache_2d is not None:
            print('loading 2d embeds_cache from args ', args.embeds_cache_2d)
            proto_dict_2d = torch.load(args.embeds_cache_2d)
        # else:
        #     if args.dataset == 'MiniImageNet':
        #         if args.backbone_class == 'ConvNet':
        #             print('loading ./embeds_cache/embeds_cache_cnn4_contrastive-init-ver1-1-corrected_2d.pt')
        #             proto_dict_2d = torch.load('./embeds_cache/embeds_cache_cnn4_contrastive-init-ver1-1-corrected_2d.pt')
        #             # print('loading ./embeds_cache/embeds_cache_cnn4_2d.pt')
        #             # proto_dict_2d = torch.load('./embeds_cache/embeds_cache_cnn4_2d.pt')
        #         elif args.backbone_class == 'Res12_ptcv':
        #             proto_dict_2d = torch.load('/home/mayug/projects/few_shot/notebooks/embeds_cache_res12_ptcv_2d.pt')
        #     elif args.dataset == 'CUB':
        #         if args.backbone_class == 'ConvNet':
        #             print('Using CUB dataset 2d embeds cache')
        #             proto_dict_2d = torch.load('/home/mayug/projects/few_shot/notebooks/embeds_cache_cnn4_2d_cub.pt')            
        self.proto_dict_2d = proto_dict_2d
        self.all_proto_2d = proto_dict_2d['embeds'].cuda()

        if self.args.mixed_precision is not None and self.args.mixed_precision!='O0':
            print('halving the embeds_cache 2d')
            self.all_proto_2d = self.all_proto_2d.half()      

        if args.embed_pool == 'max_pool':
            self.embed_pool = F.max_pool2d
        elif args.embed_pool == 'avg_pool':
            self.embed_pool = F.avg_pool2d
        elif args.embed_pool == 'post_loss_avg':
            self.embed_pool = torch.nn.Identity()
 
        print('Using embed_pool type = ', self.embed_pool)


    def save_as_numpy(self, tensor, name, debug=False):
        if self.debug:
            if isinstance(tensor, torch.Tensor):
                np_array = tensor.detach().cpu().numpy()
            else:
                np_array = tensor
            np.save(os.path.join('/home/mayug/projects/few_shot/notebooks/temp/', name+'.npy'), np_array)
 
    def get_base_protos(self, proto, ids):
        # 2d version of get_base_protos
        # querying uses 1d proto
        # but returns 3d feature maps of top_k protos

        proto = proto.squeeze()
        # print('proto before maxpooling ', proto.shape)
        if self.args.backbone_class == 'ConvNet':
            proto = F.max_pool2d(proto, kernel_size=5).squeeze()
        else:
            proto = F.adaptive_avg_pool2d(proto, output_size=(1,1)).squeeze()
        # print('proto after maxpooling ', proto.shape)

        if ids is not None:
            current_classes = list(set(ids2classes(ids)))
            remove_instances = True
        else:
            current_classes = []
            remove_instances = False
        # print('training proto, all_proto', [self.training, proto.shape, proto.min(), proto.max(), self.all_proto.shape, self.all_proto.min(), self.all_proto.max()])
        top_k, mask = get_k_base(proto, self.all_proto, k=self.args.k, remove_instances=remove_instances, 
                           all_classes=self.all_classes,
                           current_classes=current_classes,
                           train=self.training,
                           random=self.random)
        self.top_k = (top_k, mask)
    
        self.save_as_numpy(top_k, 'top_k')
        self.save_as_numpy(mask, 'mask')

        all_proto = self.all_proto_2d[~mask]

        # print('here ', all_proto.shape)

        base_protos = all_proto[top_k, :]

        # print('ttype of base protos ', [type(base_protos), base_protos.is_cuda])

        return base_protos

    def get_base_protos_fast_query(self, ids):
        # this code is wrong, need to have different code for train and test time
        # Also remove_instances of same class    
        # code is correct, it's done during the cache creation itself

        # check which of the following two statements take more time and optimize

        top_indices = np.stack([self.query_dict[id_][1][:self.args.k] for id_ in ids], axis=0)
        base_protos = self.all_proto_2d[torch.Tensor(top_indices).long()]
        
        return base_protos

    def _forward(self, instance_embs, support_idx, query_idx, ids=None, simclr_embs=None):
        # print('checking 1', [instance_embs.shape,instance_embs.min(),instance_embs.max()])
        spatial_dim = instance_embs.shape[-1]
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
        
        # proto = support.mean(dim=1) 

        proto = support.squeeze()
        self.save_as_numpy(proto, 'proto')
        # proto 1 * n_class * (5 * 5) * 64
        n_class = proto.shape[1]
        n_shot = proto.shape[0]
        k = self.args.k - 1 
        # print('proto after squeeze ', proto.shape)
        proto = proto.permute(1,0,2,3,4).reshape(-1, emb_dim, spatial_dim, spatial_dim)
        # print('after moving n_class to left most axis and reshaping ', proto.shape)
        # get topk 3d base protos is using the same 1d topk function
        # with torch.no_grad():

        if self.fast_query is None:
            # with torch.no_grad():
            base_protos = self.get_base_protos(proto, ids) # 
        else:
            base_protos = self.get_base_protos_fast_query(ids[:5*self.args.shot])
    
        # base_protos n_class * topk * 64 * (5 * 5)

        # print('base_protos', base_protos.shape)
        # print('proto before reshape', proto.shape)
        proto = proto.squeeze()
        proto = proto.reshape(proto.shape[0], emb_dim, -1).permute(0, 2, 1).contiguous()
        self.save_as_numpy(proto, 'proto_after_reshape')
        # print('proto reshaped', proto.shape)
        
        # proto n_class * (5*5) * 64

        # print('proto.shape ', proto.shape)


        base_protos = base_protos.permute(0, 2, 1, 3, 4).contiguous()
        # print('base_protos permuted ', base_protos.shape)
        # big mistake here while using fast_cache
        combined_protos = base_protos.reshape(n_class*n_shot, emb_dim, -1).permute(0, 2, 1).contiguous()
        self.save_as_numpy(combined_protos, 'combined_protos')

        # print('combined_protos.shape ', combined_protos.shape)
        # print('proto.shape ', proto.shape)
        # asd
        # print('combined_protos.shape ', combined_protos.shape)

    
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        
        # print('before slf attention proto, combined proto', [proto.shape, combined_protos.shape,combined_protos.min(),combined_protos.max()])
        if self.feat_attn==2:
            proto = self.self_attn2(proto, proto, proto)
        proto = self.slf_attn(proto, combined_protos, combined_protos)
        # print('after slf attention proto', [self.training, proto.shape, proto.min(),proto.max()])
        
        # separate to n_class and n_shot and take mean
        proto = proto.reshape(n_class, n_shot, spatial_dim*spatial_dim, emb_dim)
        # print('proto separated ', proto.shape)
        proto = proto.mean(dim=1)
        # print('proto after mean ', proto.shape)
        
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
        # asd
        # print('query after maxpool shape', [query.shape])

        if isinstance(self.embed_pool, torch.nn.Identity):
            # print('proto, query shape inside embed_pool identity')
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
            # print(['range of logits ', logits.shape, logits.min(), logits.max()])
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