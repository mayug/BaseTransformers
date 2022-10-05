from .feat import FEAT
import torch
import numpy as np
import torch.nn.functional as F


# Experimenting with different aux loss;
# transformer takes in all 80 inputs for aux loss 

class FEAT_Aux(FEAT):
    def _forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))
    
        # get mean of the support
        proto = support.mean(dim=1) # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])
    
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        # print('proto ', proto.shape)
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
            if self.args.balance==0:
                return logits, None
            aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
                                  query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            aux_task_orig = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
            aux_task = aux_task.contiguous().view(1, num_proto*(self.args.shot + self.args.query), emb_dim)
            
            # print('aux task before attn', aux_task.shape)
            
            # apply the transformation over the Aug Task
            aux_emb = self.slf_attn(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
            # compute class mean
            aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
            aux_center = torch.mean(aux_emb, 2) # T x N x d
            # print('aux_center ', aux_center.shape)
            
            if self.args.use_euclidean:
                aux_task = aux_task_orig.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
                aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
                aux_center = aux_center.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
                # print('aux center ', aux_center.shape)
                # print('aux_task ', aux_task.shape)
                # asd
                logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
            else:
                aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
                aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
    
                logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1])) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)            
            
            return logits, logits_reg            
        else:
            return logits