import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F
from model import models

from model.models import FewShotModel
from model.dataloader.mini_imagenet import MiniImageNet as Dataset
from model.dataloader.samplers import ClassSampler
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    classes = np.array([id_[:-len('00000005')] for id_ in ids])
    return classes


def get_k_base(proto, all_proto, return_weights=False, k=10, 
               train=True, remove_instances=False, all_classes=None, current_classes=None):

    
    # remove_instances, all_classes, current_classes are specifically for baseinstances case 
    mask = np.zeros(all_classes.shape).astype(np.bool)
    if train:
        start = 1
        end =0
    else:
        start = 0
        end = 1
        
    # print('all ', all_classes)
    # print('current ', current_classes)
    # print('remove_instances, train', [remove_instances, train])
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
        # print('here', mask.shape)
        # print(all_proto.shape)
    

    # print(['here', proto.shape, all_proto.shape])
    # print(['proto stats', proto.min(), proto.max()])
    # print(['all_proto stats', all_proto.min(), all_proto.max()])
    similarities = pairwise_distances_logits(proto, all_proto).squeeze()
    if similarities.dim() ==1:
        similarities = similarities.unsqueeze(0)
    similarities_sorted = torch.sort(similarities, descending=True, dim=1)

    a_ind = similarities_sorted.indices

    if return_weights:
        a = similarities_sorted.values
        a = F.softmax(a[:,start:], dim=1)
        return a_ind[:, start:k-end], a[:, :k-start-end], mask
    

    # print('top k inside ', a_ind[:, start:k-end])
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

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
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
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output
    
class FEATBaseClass2(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        if args.backbone_class == 'ConvNet':
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = 640
        elif args.backbone_class == 'Res18':
            hdim = 512
        elif args.backbone_class == 'WRN':
            hdim = 640
        else:
            raise ValueError('')
        
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)

        if args.base_protos==1: 
            # using base protos
            print('Using base protos')
            proto_dict = pickle.load(open('/home/mayug/projects/few_shot/notebooks/proto_dict_new.pkl', 'rb'))  
            self.all_proto = torch.cat([torch.Tensor(proto_dict[i]).unsqueeze(0) for i in range(len(proto_dict))], dim=0).cuda()
        elif args.base_protos==0:
            # using base instances
            print('using base instances')
            proto_dict = torch.load('/home/mayug/projects/few_shot/notebooks/embeds_cache.pt')
            self.all_proto = proto_dict['embeds'].cuda()
        self.proto_dict = proto_dict
        self.after_attn = None
        self.thresholding = False
        self.update_samples=args.update_base_samples
        self.all_classes = ids2classes(np.array(proto_dict['ids']))
        self.class_loader =  self.get_class_loader()
        self.feat_attn = args.feat_attn

    def add_base_classes(self, proto, ids):
        base_wt = self.args.base_wt
        proto = proto.squeeze()
        if ids is not None:
            current_classes = list(set(ids2classes(ids)))
            remove_instances = True
        else:
            current_classes = []
            remove_instances = False

        top_k, mask = get_k_base(proto, self.all_proto, k=self.args.k, remove_instances=remove_instances, 
                           all_classes=self.all_classes,
                           current_classes=current_classes,
                           train=self.training)
        # print('here ', [top_k.shape, wts.shape])
        # if self.thresholding:
        #     top_k_new = top_k*(wts>0.1).int()
        #     all_proto_new = torch.cat([torch.zeros(1,64).cuda(), 
        #     self.all_proto],dim=0)
        #     base_protos = all_proto_new[top_k_new, :]
        # else:
        all_proto = self.all_proto[~mask]
        base_protos = all_proto[top_k, :]
        # if base_wt==-1:
        #     # base_wt proportional to similarity
        #     modified_protos = torch.cat([(wts.unsqueeze(2).repeat(1,1,64))*(base_protos), proto.unsqueeze(1)], dim=1)
        # elif base_wt > 0:
        modified_protos = torch.cat([base_wt*(base_protos), proto.unsqueeze(1)], dim=1)
        modified_protos = modified_protos.mean(1).unsqueeze(0)
        return modified_protos

    def update_base_protos(self):
        
        class_ctr = 0
        batch_size=16
        assert self.update_samples%batch_size==0      
                
        print('updating base protos', [len(next(iter(self.class_loader))[0]), self.update_samples])
        print([len(self.all_proto), len(self.proto_dict), self.all_proto[0].mean()])

        for data, target in tqdm(self.class_loader):
            avg_embeds = np.zeros_like(self.proto_dict[0])
            
            for i in range(int(self.update_samples/batch_size)):
                with torch.no_grad():
                    embeds = self.encoder(data[i*batch_size:(i+1)*batch_size].cuda())
                    avg_embeds = avg_embeds + embeds.cpu().numpy().mean(0)
            self.proto_dict[class_ctr] = avg_embeds
            class_ctr = class_ctr + 1
        # updating all_proto
        self.all_proto = torch.cat([torch.Tensor(self.proto_dict[i]).unsqueeze(0) for i in range(len(self.proto_dict))], dim=0).cuda()
        

    def get_class_loader(self):
        trainset = Dataset('train', self.args)
        class_sampler = ClassSampler(trainset.label, n_per=self.update_samples)
        train_loader_class = DataLoader(trainset, batch_sampler=class_sampler)
        return train_loader_class

        

    def _forward(self, instance_embs, support_idx, query_idx, ids=None):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))
    
        
        

        # get mean of the support
        proto = support.mean(dim=1) # Ntask x NK x d

        # add closest 10 base classes to proto and calculate mean
        proto = self.add_base_classes(proto, ids)
        

        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])
    
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)

        #exp removing transformer
        if self.feat_attn==1:
            proto = self.slf_attn(proto, proto, proto) 
          
        self.after_attn = proto
        
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
            return logits, None
            aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
                                  query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
            # apply the transformation over the Aug Task
            aux_emb = self.slf_attn(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
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
