from .feat_basetransformer3_2d import FEATBaseTransformer3_2d
from .feat_basetransformer3 import FEATBaseTransformer3
from .feat_basetransformer2 import get_k_base, ids2classes, ScaledDotProductAttention
import torch
import numpy as np
import torch.nn.functional as F
import os
import torch.nn as nn


class MultiHeadAttention_query(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, debug=False):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False) 
        # w_qs not necessary as the residual need not be embedded,
        # w_ks same as w_vs for query(protos) and key(combined_protos)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm_query = nn.LayerNorm(d_model)

        # self.fc = nn.Linear(n_head * d_v, d_model)
        self.query_v = nn.Linear(d_model, n_head * d_v)
        nn.init.xavier_normal_(self.query_v.weight)
        # self.fc_query = nn.Linear(n_head * d_v, d_model)
        # nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        self.debug = debug

    def save_as_numpy(self, tensor, name, debug=False):
        if self.debug:
            if isinstance(tensor, torch.Tensor):
                np_array = tensor.detach().cpu().numpy()
            else:
                np_array = tensor
            np.save(os.path.join('/home/mayug/projects/few_shot/notebooks/temp/', name+'.npy'), np_array)
        
    def forward(self, q, k, v, query, query_base_protos):
        # print('here', [q.shape, k.shape, v.shape])
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        

        residual = self.query_v(q)
        # print('residual ', residual.shape)

        q = self.w_ks(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        # print('Inside multi head before self.attention ', [q.shape, k.shape, v.shape])

        output, attn, log_attn = self.attention(q, k, v)
        self.save_as_numpy(attn, 'attn')
        self.save_as_numpy(log_attn, 'log_attn')
        # print('here ', attn.shape)
        # print('log attn ', log_attn.shape)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        # print(output.shape)
        # asd
        # output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        
        # now same for query instance

        sz_b, len_q_query, _ = query.size()
        sz_b, len_k_query, _ = query_base_protos.size()
        sz_b, len_v_query, _ = query_base_protos.size()

        residual_query = self.query_v(query)
        # print('residual_query' , residual_query.shape)

        q_query = self.w_ks(query).view(sz_b, len_q_query, n_head, d_k)
        k_query = self.w_ks(query_base_protos).view(sz_b, len_k_query, n_head, d_k)
        v_query = self.w_vs(query_base_protos).view(sz_b, len_v_query, n_head, d_v)
        
        q_query = q_query.permute(2, 0, 1, 3).contiguous().view(-1, len_q_query, d_k) # (n*b) x lq x dk
        k_query = k_query.permute(2, 0, 1, 3).contiguous().view(-1, len_k_query, d_k) # (n*b) x lk x dk
        v_query = v_query.permute(2, 0, 1, 3).contiguous().view(-1, len_v_query, d_v) # (n*b) x lv x dv

        # print('q_query, k_query, v_query'[])

        output_query, attn_query, log_attn_query = self.attention(q_query, k_query, v_query)

        output_query = output_query.view(n_head, sz_b, len_q_query, d_v)
        output_query = output_query.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q_query, -1) # b x lq x (n*dv)

        # output_query = self.dropout(self.fc_query(output_query))


        # output_query = self.dropout(self.fc(output_query))

        output_query = self.layer_norm_query(output_query + residual_query)

        # print('output ', [output.shape, output_query.shape])

        return output, output_query

    
def load_numpy(name):
    return np.load(os.path.join('/home/mayug/projects/few_shot/notebooks/temp/', name+'.npy'))


class FEATBaseTransformer3_2d_sym(FEATBaseTransformer3_2d):
    def __init__(self, args):
        args.max_pool = None
        args.resize = False
        super().__init__(args)
        # these 2d embeddings of base instances are used for combination

        # if args.dataset == 'MiniImageNet':
        #     if args.backbone_class == 'ConvNet':
        #         proto_dict_2d = torch.load('/home/mayug/projects/few_shot/notebooks/embeds_cache_cnn4_2d.pt')
        #     elif args.backbone_class == 'Res12_ptcv':
        #         proto_dict_2d = torch.load('/home/mayug/projects/few_shot/notebooks/embeds_cache_res12_ptcv_2d.pt')
        # elif args.dataset == 'CUB':
        #     if args.backbone_class == 'ConvNet':
        #         print('Using CUB dataset 2d embeds cache')
        #         proto_dict_2d = torch.load('/home/mayug/projects/few_shot/notebooks/embeds_cache_cnn4_2d_cub.pt')            
        # self.proto_dict_2d = proto_dict_2d
        # self.all_proto_2d = proto_dict_2d['embeds'].cuda()
        
        # if args.embed_pool == 'max_pool':
        #     self.embed_pool = F.max_pool2d
        # elif args.embed_pool == 'avg_pool':
        #     self.embed_pool = F.avg_pool2d
        # elif args.embed_pool == 'post_loss_avg':
        #     self.embed_pool = torch.nn.Identity()
 
        # print('Using embed_pool type = ', self.embed_pool)
        # hdim = 64
        self.slf_attn = MultiHeadAttention_query(args.n_heads, self.hdim, 
            self.hdim, self.hdim, dropout=0.5)


    # def save_as_numpy(self, tensor, name, debug=False):
    #     if self.debug:
    #         if isinstance(tensor, torch.Tensor):
    #             np_array = tensor.detach().cpu().numpy()
    #         else:
    #             np_array = tensor
    #         np.save(os.path.join('/home/mayug/projects/few_shot/notebooks/temp/', name+'.npy'), np_array)
 
    # def get_base_protos(self, proto, ids):
    #     # 2d version of get_base_protos
    #     # querying uses 1d proto
    #     # but returns 3d feature maps of top_k protos

    #     proto = proto.squeeze()
    #     # print('proto before maxpooling ', proto.shape)
    #     proto = F.max_pool2d(proto, kernel_size=5).squeeze()
    #     # print('proto after maxpooling ', proto.shape)

    #     if ids is not None:
    #         current_classes = list(set(ids2classes(ids)))
    #         remove_instances = True
    #     else:
    #         current_classes = []
    #         remove_instances = False

    #     top_k, mask = get_k_base(proto, self.all_proto, k=self.args.k, remove_instances=remove_instances, 
    #                        all_classes=self.all_classes,
    #                        current_classes=current_classes,
    #                        train=self.training,
    #                        random=self.random)
    #     self.top_k = (top_k, mask)
    
    #     self.save_as_numpy(top_k, 'top_k')
    #     self.save_as_numpy(mask, 'mask')

    #     all_proto = self.all_proto_2d[~mask]


    #     base_protos = all_proto[top_k, :]

    #     # print('ttype of base protos ', [type(base_protos), base_protos.is_cuda])

    #     return base_protos


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
        proto = support.mean(dim=1) 
        self.save_as_numpy(proto, 'proto')
        # proto 1 * n_class * (5 * 5) * 64
        n_class = proto.shape[1]
        n_batch = proto.shape[0]
        k = self.args.k - 1

        query = query.view(-1, emb_dim, spatial_dim, spatial_dim)
        # print('proto after mean ', [proto.shape, query.shape])
        n_query = query.shape[0]
        # print('proto after mean ', proto.shape)

        # get topk 3d base protos is using the same 1d topk function
        if self.fast_query is None:
            with torch.no_grad():
                base_protos = self.get_base_protos(proto, ids) # 
        else:
            base_protos = self.get_base_protos_fast_query(ids[:5])

        if self.fast_query is None:
            # with torch.no_grad():
            base_protos_query = self.get_base_protos(query.squeeze(), ids) # 
        else:
            # base_protos_query = self.get_base_protos_fast_query(ids[5:])

            # for base_protos_query we don't know the ids of query, so for each query
            # we can only use it's features for getting closest base_instances
            # or we can get closest base instances of all suppor examples in the 
            # episode

            # base_protos_query = self.get_base_protos_fast_query(ids[:5]*self.args.query)

            # to simulate getting closest base_instances of all support examples
            # shuffle the query*top_k axis

            k=5
            base_protos_query = self.get_base_protos_fast_query(ids[:5]*self.args.query, k)

            base_protos_query = base_protos_query.reshape(-1, 64, 5, 5)
            shuffle = torch.randint(low=0, high=75*30, size=(75*30,))
            # print(shuffle.shape)
            base_protos_query = base_protos_query[shuffle, :, :, :]
            base_protos_query = base_protos_query.reshape(75, 30, 64, 5, 5)


        # base_protos = self.get_base_protos(proto, ids) # 
        # base_protos_query = self.get_base_protos(query, ids)
        self.save_as_numpy(base_protos, 'base_protos')
        # base_protos n_class * topk * 64 * (5 * 5)

        # print('base_protos_query', base_protos_query.shape)
        # asd
        # print('proto ', proto.shape)
        # proto = proto.reshape(proto.shape[1], -1, emb_dim)
        proto = proto.reshape(proto.shape[1], emb_dim, -1).permute(0, 2, 1).contiguous()
        query = query.reshape(query.shape[0], emb_dim, -1).permute(0, 2, 1).contiguous()
        self.save_as_numpy(proto, 'proto_after_reshape')
        # print('proto reshaped', proto.shape)
        # proto n_class * (5*5) * 64

        # print('proto.shape ', proto.shape)

        # combined_protos = base_protos.reshape(n_class*n_batch, k*num_patches, emb_dim)

        base_protos = base_protos.permute(0, 2, 1, 3, 4).contiguous()
        combined_protos = base_protos.reshape(n_class*n_batch, emb_dim, -1).permute(0, 2, 1).contiguous()
        
        base_protos_query = base_protos_query.permute(0, 2, 1, 3, 4).contiguous()
        combined_protos_query = base_protos_query.reshape(n_query*n_batch, emb_dim, -1).permute(0, 2, 1).contiguous()
        
        self.save_as_numpy(combined_protos, 'combined_protos')

        # print('combined_protos.shape ', combined_protos.shape)
        # print('proto.shape ', proto.shape)
        # print('combined_protos.shape ', combined_protos.shape)

    
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        
        # print('before slf attention proto, combined proto', [proto.shape, combined_protos.shape,combined_protos.min(),combined_protos.max()])
        
        if self.feat_attn==2:
            proto = self.self_attn2(proto, proto, proto)
        proto, query = self.slf_attn(proto, combined_protos,
            combined_protos, query, combined_protos_query)
        # print('after slf attention proto', [self.training, proto.shape, proto.min(),proto.max()])

        # print('after slf attention proto', [proto.shape])
        proto = proto.permute(0, 2, 1).contiguous()
        # print('proto after permute nad before making 2d again ', proto.shape)
        
        proto = proto.view(-1, emb_dim, spatial_dim, spatial_dim)
        # print('proto after making 2d and before maxpool ', proto.shape)

        # mean for proto; unhide below
        # proto = proto.mean(dim=1).unsqueeze(0)
  
        # print('after maxpool proto', [proto.shape])
        # print('query before view operation', query.shape)
        query = query.permute(0, 2, 1).contiguous()
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

        # print('before loss  ', [proto.shape, query.shape])
        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
    
            logits = - torch.mean((proto - query) ** 2, 2) / self.args.temperature
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
# class FEATBaseTransformer3_2d_sym(FEATBaseTransformer3_2d):
#     def __init__(self, args):
#         args.max_pool = None
#         args.resize = False
#         super().__init__(args)
#         # these 2d embeddings of base instances are used for combination
#         self.slf_attn = MultiHeadAttention_query(args.n_heads, self.hdim, 
#             self.hdim, self.hdim, dropout=0.5)
        

#     def _forward(self, instance_embs, support_idx, query_idx, ids=None):
#         # print('checking 1', [instance_embs.shape,instance_embs.min(),instance_embs.max()])
#         spatial_dim = instance_embs.shape[-1]
#         self.save_as_numpy(instance_embs, 'instance_embs')

#         emb_dim = instance_embs.size(-3)
#         num_patches = np.prod(instance_embs.shape[-2:]) 

#         # print("emb_dim, num_patches", [emb_dim, num_patches])
#         # print('instance_embs', instance_embs.shape)
#         # print('support_idx, query_idx', [support_idx.shape, query_idx.shape])
#         # organize support/query data

#         # print(instance_embs.shape)
#         support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (emb_dim, spatial_dim, spatial_dim,)))
#         query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (emb_dim, spatial_dim, spatial_dim,)))
    
#         # print('support, query', [support.shape, query.shape])
#         self.save_as_numpy(support, 'support')
#         self.save_as_numpy(support, 'query')
#         # support n_class * K * (5 * 5) * 64
#         # get mean of the support
#         proto = support.mean(dim=1) 
#         self.save_as_numpy(proto, 'proto')
#         # proto 1 * n_class * (5 * 5) * 64
#         n_class = proto.shape[1]
#         n_batch = proto.shape[0]
#         k = self.args.k - 1 

#         query = query.view(-1, emb_dim, spatial_dim, spatial_dim)
#         # print('proto after mean ', [proto.shape, query.shape])
#         n_query = query.shape[0]
#         # get topk 3d base protos is using the same 1d topk function
#         # with torch.no_grad():
        
#         base_protos = self.get_base_protos(proto, ids) # 

#         base_protos_query = self.get_base_protos(query, ids)

#         # print('base_protos_query ', [base_protos.shape, base_protos_query.shape])
        

#         self.save_as_numpy(base_protos, 'base_protos')
#         # base_protos n_class * topk * 64 * (5 * 5)

#         # print('base_protos', base_protos.shape)
#         # print('proto ', proto.shape)
#         proto = proto.squeeze()
#         # proto = proto.reshape(proto.shape[1], -1, emb_dim)
#         proto = proto.reshape(proto.shape[0], emb_dim, -1).permute(0, 2, 1).contiguous()
#         query = query.reshape(query.shape[0], emb_dim, -1).permute(0, 2, 1).contiguous()
#         # print('proto after reshape ', [proto.shape, query.shape])
#         self.save_as_numpy(proto, 'proto_after_reshape')
#         # print('proto reshaped', proto.shape)
#         # proto n_class * (5*5) * 64

#         # print('proto.shape ', proto.shape)

#         # combined_protos = base_protos.reshape(n_class*n_batch, k*num_patches, emb_dim)
        
#         base_protos = base_protos.permute(0, 2, 1, 3, 4).contiguous()
#         combined_protos = base_protos.reshape(n_class*n_batch, emb_dim, -1).permute(0, 2, 1).contiguous()
#         base_protos_query = base_protos_query.permute(0, 2, 1, 3, 4).contiguous()
#         combined_protos_query = base_protos_query.reshape(n_query*n_batch, emb_dim, -1).permute(0, 2, 1).contiguous()
#         self.save_as_numpy(combined_protos, 'combined_protos')

#         # print('combined_protos.shape ', [combined_protos.shape, combined_protos_query.shape])
        
#         # print('proto.shape ', proto.shape)
#         # print('combined_protos.shape ', combined_protos.shape)

    
#         # query: (num_batch, num_query, num_proto, num_emb)
#         # proto: (num_batch, num_proto, num_emb)
        
#         # print('before slf attention proto, combined proto', [proto.shape, combined_protos.shape,combined_protos.min(),combined_protos.max()])
        
#         if self.feat_attn==2:
#             proto = self.self_attn2(proto, proto, proto)
#         proto, query = self.slf_attn(proto, combined_protos,
#             combined_protos, query, combined_protos_query)
#         # print('after slf attention proto', [self.training, proto.shape, proto.min(),proto.max()])
#         # print('after slf attention query', [self.training, query.shape, query.min(),query.max()])
#         # print('after slf attention proto', [proto.shape])
#         # asd
#         proto = proto.permute(0, 2, 1).contiguous()
        
#         proto = proto.view(-1, emb_dim, spatial_dim, spatial_dim)
        
#         query = query.permute(0, 2, 1).contiguous()
        
#         query = query.view(-1, emb_dim, spatial_dim, spatial_dim)

#         if isinstance(self.embed_pool, torch.nn.Identity):
#             # print(proto.shape)
#             # print(query.shape)
#             proto = proto.reshape(proto.shape[0], -1).unsqueeze(0)
#             query = query.reshape(query.shape[0], -1)
#             # print(proto.shape)
#             # print(query.shape)
#             emb_dim = emb_dim*(spatial_dim**2)
          
#             # print(proto.shape)
#             # print(query.shape)
#             # asd

#             # asd
#         else:
#             proto = self.embed_pool(proto, kernel_size=5).squeeze().unsqueeze(0)
#             query = self.embed_pool(query, kernel_size=5).squeeze()
       
            
  

#         num_batch = proto.shape[0]
#         num_proto = proto.shape[1]
#         num_query = np.prod(query_idx.shape[-2:])
#         # print('after slf attention proto', [proto.shape])

#         if self.feat_attn==1:
#             proto = self.self_attn2(proto, proto, proto)

#         self.after_attn = proto

#         # print('before loss  ', [proto.shape, query.shape])
        
#         if self.args.use_euclidean:
#             query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
#             proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
#             proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
    
#             logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
#             # print('check logits ', logits.shape)
#         else:
#             proto = F.normalize(proto, dim=-1) # normalize for cosine distance
#             query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

#             logits = torch.bmm(query, proto.permute([0,2,1]).contiguous()) / self.args.temperature
#             logits = logits.view(-1, num_proto)
        
#         # for regularization
#         if self.training:
#             # return logits, None
#             # TODO this can be further adapted for basetransformer version
#             if self.args.balance==0:
#                 return logits, None
#             aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
#                                   query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
#             num_query = np.prod(aux_task.shape[1:3])
#             aux_task = aux_task.permute([0, 2, 1, 3])
#             aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
#             # apply the transformation over the Aug Task
#             if self.feat_attn==1:
#                 aux_emb = self.self_attn2(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
#             else:
#                 # print('before aux forward pass ', [aux_task.shape, combined_protos.shape])
#                 aux_emb = self.slf_attn(aux_task, combined_protos, combined_protos)
#                 # print('after aux forward pass ', aux_emb.shape)
#             # compute class mean
#             aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
#             aux_center = torch.mean(aux_emb, 2) # T x N x d
            
#             if self.args.use_euclidean:
#                 aux_task = aux_task.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
#                 aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
#                 aux_center = aux_center.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
#                 # print('aux center ', aux_center.shape)
#                 # print('aux_task ', aux_task.shape)
#                 logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
#             else:
#                 aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
#                 aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
    
#                 logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1]).contiguous) / self.args.temperature2
#                 logits_reg = logits_reg.view(-1, num_proto)            
            
#             return logits, logits_reg            
#         else:
#             return logits 