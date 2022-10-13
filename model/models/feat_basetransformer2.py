import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F
from model import models
import os
import os.path as osp
from model.models import FewShotModel


class pca():    
    def __init__(self, pca_dim, q, whiten=False):
        self.v = None
        self.s= None
        self.pca_dim = pca_dim
        self.q = q
        self.whiten = whiten
        # s- singular values= root of eigen values
        # to whiten divide by root ofeigen values
        # therefore divide by s
    def normalized_pca(self, a):
    
        # x has to be n_samples*embedding_dim vector
        n_samples = a.shape[0]
#         n_samples=1
        a_normalized = F.normalize(a, dim=1)
        if self.v is None:
            print('Making v')
            u,s,v = torch.pca_lowrank(a_normalized, self.q)
            self.v = v
            self.s = s
        else:
            print('Using previous v')
        a_pca = torch.matmul(a_normalized, self.v[:, :self.pca_dim])
        if self.whiten:
            
            a_pca = np.sqrt(n_samples) * (a_pca / self.s[:self.pca_dim])
        
        a_pca_normalized = F.normalize(a_pca, dim=1)
        return a_pca_normalized


def pairwise_distances_logits(query, proto, distance_type='euclid'):
    #  query n * dim
    #  prototype n_c * dim
    if distance_type == 'euclid':
        n = query.shape[0]
        m = proto.shape[0]
        distances = -((query.unsqueeze(1).expand(n, m, -1) -
                   proto.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)  
        
#         print(distances.shape)
        return distances
    elif distance_type == 'cosine':
        emb_dim = proto.shape[-1]
        proto = F.normalize(proto, dim=-1) # normalize for cosine distance
        query = query.view(-1, emb_dim) # (Nbatch,  Nq*Nw, d)
        
        # print([query.shape, proto.shape])
        logits = torch.matmul(query, proto.transpose(1,0))
#         print('logits.shape', logits.shape)
        return logits


def ids2classes(ids):
    # print('use object function')
    # raise NotImplementedError
    classes = np.array([id_[:-len('00000005')] for id_ in ids])
    return classes


def get_k_base(proto, all_proto, return_weights=False, k=10, 
               train=True, remove_instances=False, all_classes=None, current_classes=None,
               random=False):


    
    # remove_instances, all_classes, current_classes are specifically for baseinstances case 
    mask = np.zeros(all_classes.shape).astype(np.bool)

    # why this bit of code is required ?? while training in baseinstances case
    # in base_protos case this makes sense as you don't want itself to be in the combined_protos


    # all instances of the 5 classes are removed by remove_instances;
    if train and not remove_instances:
    # if train:
        start = 1
        end =0
    else:
        start = 0
        end = 1
        

    if random:
        mask = mask + 1
        a_ind = torch.randint(low=0, high=all_classes.shape[0], size=[proto.shape[0], k-1])
        return a_ind, mask


    if remove_instances:
        # remove all instances of the current class from all_proto
        filtered_ids = []
        for curr in current_classes:
            filtered_ids.append((np.argwhere(curr==all_classes)).squeeze())
        
        filtered_ids = np.concatenate(filtered_ids)
        mask = np.zeros(all_classes.shape).astype(np.bool)
 
        mask[filtered_ids] = 1

        all_proto = all_proto[~mask]
#         all_proto[mask] = 100
        
        start = 0
        end = 1 

    


    similarities = pairwise_distances_logits(proto, all_proto).squeeze()
    if similarities.dim() ==1:
        similarities = similarities.unsqueeze(0)
    similarities_sorted = torch.sort(similarities, descending=True, dim=1)

    a_ind = similarities_sorted.indices

    if return_weights:
        a = similarities_sorted.values
        a = F.softmax(a[:,start:], dim=1)
        return a_ind[:, start:k-end], a[:, :k-start-end], mask
    

    return a_ind[:, start:k-end], mask




class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, debug=False):
        super().__init__()
        print('Creating transformer with d_k, d_v, d_model = ', [d_k, d_v, d_model])
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        self.debug = debug

    def save_as_numpy(self, tensor, name, debug=False):
        if self.debug:
            if isinstance(tensor, torch.Tensor):
                np_array = tensor.detach().cpu().numpy()
            else:
                np_array = tensor
            np.save(os.path.join('/home/mayug/projects/few_shot/notebooks/temp/', name+'.npy'), np_array)
        
    def forward(self, q, k, v):
        # print('here', [q.shape, k.shape, v.shape])
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # print('here inside self attn q ', [type(q.detach().cpu().numpy()[0,0,0]), q.shape, q[0,0,0]])
        # print('here inside self attn k ', [type(k.detach().cpu().numpy()[0,0,0]), k.shape, k[0,0,0]])
        # print('here inside self attn v ', [type(v.detach().cpu().numpy()[0,0,0]), v.shape, v[0,0,0]])
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
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

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output
    
class FEATBaseTransformer2(FewShotModel):
    def __init__(self, args):
        max_pool = args.max_pool
        resize = args.resize
        self.embeds_cache_root = './embeds_cache/'
        super().__init__(args, max_pool=max_pool, resize = resize)
        if args.backbone_class == 'ConvNet':
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = args.dim_model
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
        elif args.backbone_class == 'WRN':
            hdim = 640
        elif args.backbone_class == 'Res12_ptcv':
            hdim = 512
        else:
            raise ValueError('')
        
        if args.tx_k_v:
            tx_k_v = args.tx_k_v
        else:
            tx_k_v = hdim
        print('Creating slf_attn with hdim, tx_k_v = ', [hdim, tx_k_v])
        self.slf_attn = MultiHeadAttention(args.n_heads, hdim, tx_k_v, tx_k_v, dropout=0.5)
        self.self_attn2 = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        proto_dict = None
        self.fast_query = args.fast_query
        if args.base_protos==1: 
            asd
            # using base protos
            print('Using base protos')
            proto_dict = pickle.load(open('/home/mayug/projects/few_shot/notebooks/proto_dict_new.pkl', 'rb'))  
            self.all_proto = torch.cat([torch.Tensor(proto_dict[i]).unsqueeze(0) for i in range(len(proto_dict))], dim=0).cuda()
        elif args.base_protos==0:
            
            # using base instances
            print('using base instances')
            print('backbone_class ', [args.backbone_class, args.backbone_class=='Res12_ptcv', args.dataset])
            if self.fast_query is None:
                if args.embeds_cache_1d is not None:
                    print('loading 1d embeds_cache from args ', args.embeds_cache_1d)
                    proto_dict = torch.load(args.embeds_cache_1d)
   
                self.all_proto = proto_dict['embeds'].cuda() 
                if self.args.mixed_precision is not None and self.args.mixed_precision!='O0':
                    print('halving the embeds_cache 1d')
                    self.all_proto = self.all_proto.half()
                self.proto_dict = proto_dict
                self.all_classes = self.ids2classes(np.array(proto_dict['ids']))
        if self.fast_query is not None:
            print('Loading fast query_dict ', self.fast_query)
            print('path is ', self.fast_query)
            self.query_dict = torch.load(self.fast_query)
            # self.all_base_ids = np.array(list(self.query_dict.keys()))[:38400]
        self.after_attn = None
        self.feat_attn = args.feat_attn
        self.top_k = None

        self.pca_dim = args.pca_dim
        if self.pca_dim is not None:
            print('Using PCA dim ', self.pca_dim)
            self.v = None

        self.debug = bool(args.debug)
        self.random = bool(args.random)

        self.remove_instances = bool(self.args.remove_instances)
        

        if self.args.query_model_path is not None:
            self.query_model = self.args.query_model_path
            print('Using query_model for querying ', args.query_model_path)
            # self.query_model = torch.load(self.args.query_model_path)
            proto_dict = torch.load(self.query_model)
            self.all_proto_new = proto_dict['embeds'].cuda().half()
            self.proto_dict_all = torch.load(self.query_model[:-len('.pt')]+'_all.pt')
            # proto_dict = torch.load('/home/mayug/projects/few_shot/notebooks/embeds_cache_r12_imagenet.pt')
            # self.all_proto_new = proto_dict['embeds'].cuda()
            # self.proto_dict_all = torch.load('/home/mayug/projects/few_shot/notebooks/embeds_cache_r12_imagenet_all.pt')

    def normalized_pca(self, a, pca_dim, q=300):
    
        # x has to be n_samples*embedding_dim vector
        a_normalized = F.normalize(a, dim=1)
        if self.v is None:
            u,s,v = torch.pca_lowrank(a_normalized, q)
            self.v = v
        a_pca = torch.matmul(a_normalized, self.v[:, :self.pca_dim])
        a_pca_normalized = F.normalize(a_pca, dim=1)
        return a_pca_normalized
    
    def ids2classes(self, ids):
        if self.args.dataset=='MiniImagenet' or self.args.dataset=='TieredImageNet_og':
            classes = np.array([id_[:-len('00000005')] for id_ in ids])
            return classes
        elif self.args.dataset=='CUB':
            classes = np.array(['_'.join(id_.split('_')[:-2]) for id_ in ids])
        else:
            raise NotImplementedError
        return classes
            

    def get_proto_new(self, ids):
        # print('calling get proto new')
        # using presaved val and test protos instead of running through resnet_imagenet each time
        # print('using get proto new')
        indices = [np.argwhere(np.array(self.proto_dict_all['ids'])==i).item() for i in ids]
        
        return torch.cat([self.proto_dict_all['embeds'][i].unsqueeze(0) for i in indices], dim=0).cuda()

    def get_base_protos(self, proto, ids):


        proto = proto.squeeze()
        all_proto = self.all_proto
        # print('here self traiing ', self.training)
        # print('ids ', ids)
        if ids is not None:
            current_classes = list(set(self.ids2classes(ids)))
            # remove_instances = True
        else:
            current_classes = []
            # remove_instances = False
        # print('current classes ', current_classes)
        # print('proto 5 mean', proto[5].mean())
        if self.args.query_model_path is not None:
            
            proto = self.get_proto_new(ids[:5])
            all_proto = self.all_proto_new


        if self.pca_dim is not None:
            all_proto = self.normalized_pca(all_proto)
            proto = self.normalized_pca(proto)


        # print('before get_k_base ', [proto.shape, all_proto.shape])
        
        top_k, mask = get_k_base(proto, all_proto, k=self.args.k,
                        remove_instances=self.remove_instances, 
                        all_classes=self.all_classes,
                        current_classes=current_classes,
                        train=self.training, random=self.random)

        self.top_k = (top_k, mask)
        
        all_proto = self.all_proto[~mask]

        base_protos = all_proto[top_k, :]

        return base_protos

    def _forward(self, instance_embs, support_idx, query_idx, ids=None):
        print('here inside bit2')
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))
    
        
        

        # get mean of the support
        proto = support.mean(dim=1) # Ntask x NK x d

        # add closest 10 base classes to proto and calculate mean
        with torch.no_grad():
            base_protos = self.get_base_protos(proto, ids)
        
        
        # including base_protos into key and value
        proto = proto.squeeze()

        combined_protos = base_protos

        proto = proto.unsqueeze(0)

        combined_protos = combined_protos.reshape(-1, emb_dim).unsqueeze(0)

        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])
    
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        
        proto = self.slf_attn(proto, combined_protos, combined_protos)

        if self.feat_attn==1:
            proto = self.self_attn2(proto, proto, proto)
        self.after_attn = proto

        # print('after attention ', proto.shape)
        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else:
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)
        
        # for regularization
        if self.training:
            # return logits, None
            # TODO this can be further adapted for basetransformer version

            aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
                                  query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
            # apply the transformation over the Aug Task
            if self.feat_attn==1:
                aux_emb = self.self_attn2(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
            else:
                aux_emb = self.slf_attn(aux_task, aux_task, aux_task)
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
    
                logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1])) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)            
            
            return logits, logits_reg            
        else:
            return logits   
