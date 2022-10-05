import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel


def pairwise_distances_logits(a, b):
    if b.ndim==2:
        n = a.shape[0]
        m = b.shape[0]
        logits = -((a.unsqueeze(1).expand(n, m, -1) -
                    b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    elif b.ndim==3:
        n = a.shape[0]
        m = b.shape[1]
        # print('a, b', a.shape, b.shape)
        logits = -(((a.unsqueeze(1).expand(n, m, -1) -
                    b)**2).sum(dim=2))
    
    return logits

class QueryGatedSelfAttention(nn.Module):
    ''' Query gated attention + self attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, query, query_q, query_k, query_v):

        n_dim = query.shape[-1]
        n_query = query.shape[1]
        n_proto = q.shape[1]

        
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        
        query_attn = torch.bmm(query_q, query_k.transpose(1, 2))
        query_attn = query_attn / self.temperature
        query_attn = self.softmax(query_attn)
        query_attn = self.dropout(query_attn)
        
        query_attn = query_attn.unsqueeze(-1).repeat(1, 1, 1, n_dim)
        
        output = output.unsqueeze(1).repeat(1, n_query, 1, 1)
        
        v_rep = v.unsqueeze(1).repeat(1, n_query, 1, 1)
        
        # print('attn, query attn, output v_rep', [attn.shape, query_attn.shape, output.shape, v_rep.shape])
        query_v = query_v.unsqueeze(2).repeat(1, 1, n_proto, 1)
        
        # output = output + query_attn*v_rep + query_attn*query_v   
        
        output = query_attn*v_rep + query_attn*query_v  
        # output = output + query_attn*v_rep
        
        
        return output, attn, log_attn


class MultiHeadQuery_selfattention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, n_class=5, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.w_query = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_query_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_query_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_query_v = nn.Linear(d_model, n_head * d_k, bias=False)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.w_query.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.w_query_q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.w_query_k.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.w_query_v.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = QueryGatedSelfAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout1 = nn.Dropout(dropout)
        
        # self.classifier = nn.Linear(d_model, n_class)
        # nn.init.xavier_normal_(self.fc.weight)
        # self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, q, k, v, query, query_q, query_k, query_v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        sz_b, len_query, _ = query.size()
        
        residual = q.unsqueeze(1).repeat(1, len_query, 1, 1)
        # residual = torch.cat(k,q)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        query = self.w_query(query).view(sz_b, len_query, n_head, d_k)

        query_q = self.w_query_q(query_q).view(sz_b, len_query, n_head, d_k)
        query_k = self.w_query_k(query_k).view(sz_b, len_k, n_head, d_k)
        query_v = self.w_query_v(query_v).view(sz_b, len_query, n_head, d_k)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, len_query, d_k) # (n*b) x lquery x dk

        query_q = query_q.permute(2, 0, 1, 3).contiguous().view(-1, len_query, d_k) # (n*b) x lquery x dk
        query_k = query_k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        query_v = query_v.permute(2, 0, 1, 3).contiguous().view(-1, len_query, d_k)

#         print('q,k,v,query ', [q.shape, k.shape, v.shape, query.shape])
        output, attn, log_attn = self.attention(q, k, v, query, query_q, query_k, query_v)
#         print('output.shape', output.shape)
        output = output.view(n_head, sz_b, len_query, len_k, d_v)
#         print('output.shape', output.shape)
        output = output.permute(1, 2, 3, 0, 4).contiguous().view(sz_b, len_query, len_k, -1) # b x lq x (n*dv)
#         print('output.shape', output.shape)
        output = self.dropout1(self.fc(output))
#         print('after fc', output.shape)
#         print('residual ', residual.shape)


        output = self.layer_norm(output + residual)
        # output = 
        # output = self.dropout2(self.classifier(output))
        return output
    
class FEATv3_2_2(FewShotModel):
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
        
        self.slf_attn = MultiHeadQuery_selfattention(1, hdim, hdim, hdim, dropout=0.5)          
    def _forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)
        # print('support idx ', support_idx)
        # print('query_idx, ', query_idx)
        # organize support/query data

        # support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        # query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))
       
       # this is only for 5way 1 shot
        ways = 5
        shot = 1
        query_num = 15
        support = instance_embs[support_idx.view(-1)]
        query = instance_embs[query_idx.view(-1)]

        # print('support ', support.shape)
        # print('query ', query.shape)

        support = support.unsqueeze(0)
        query = query.unsqueeze(0)

        # print('support ', support.shape)
        # print('query ', query.shape)

        # get mean of the support
        proto = support.reshape(ways, shot, emb_dim).mean(dim=1) # Ntask x NK x d
        # num_batch = proto.shape[0]
        # num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)

        # print('proto shape', proto.shape)
        # print('query shape', query.shape)
        
        proto = proto.unsqueeze(0)

        # query = query.view(-1, emb_dim).unsqueeze(0) # (Nbatch*Nq*Nw, 1, d)
        # print('query shape', query.shape)

        # print('proto shape', proto.shape)
        # print('query shape', query.shape) 

        proto = self.slf_attn(proto, proto, proto, query, query, proto, query)

        # print('proto shape', proto.shape)
             


                
        if self.args.use_euclidean:

            proto = proto.squeeze()
            query = query.squeeze()
            # print('proto shape', proto.shape)

            # print('query shape', query.shape)  
            
            logits = pairwise_distances_logits(query, proto)/self.args.temperature

            # print('logits ', logits.shape)
            # proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            # proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            # logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
 
        
        # for regularization
        if self.training:
            aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
                                  query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
            
            # print('instance_embs', instance_embs.shape)
            embed_size = instance_embs.shape[-1]
            aux_query = instance_embs.unsqueeze(0).repeat(ways, 1, 1)
            
            # print('aux task ', aux_task.shape)
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            # print('aux task ', aux_task.shape)
            aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)

            # # apply the transformation over the Aug Task
            aux_emb = self.slf_attn(aux_task, aux_task, aux_task, aux_query, aux_query, aux_task, aux_query) # T x N x (K+Kq) x d
            # # compute class mean
            # print('aux_emb ', aux_emb.shape)
            # num_batch = 1
            # aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
            aux_center = torch.mean(aux_emb, 2) # T x N x d
            aux_center = aux_center.permute(1, 0, 2)

            # print('aux task ', aux_task.shape)

            aux_task = aux_task.permute(1, 0, 2).reshape(-1, embed_size)

            # print('aux task ', aux_task.shape)
            # print('aux_center ', aux_center.shape)

            logits_reg = pairwise_distances_logits(aux_task, aux_center)/self.args.temperature2
            
            return logits, logits_reg            
        else:
            return logits   
