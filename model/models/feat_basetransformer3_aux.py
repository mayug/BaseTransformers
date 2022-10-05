from .feat_basetransformer3 import FEATBaseTransformer3
import torch
import numpy as np
import torch.nn.functional as F


# BaseTransformer3 --> folds the prorotypes into batch dimension
# so attention between 1 prototype and its topk base instances

# auxiliary loss experiments; 
# auxs loss 2 to make sure k+kq instances after adaptation are close to its baseinstances and away from other class base instances


class FEATBaseTransformer3_Aux(FEATBaseTransformer3):
    def _forward(self, instance_embs, support_idx, query_idx, ids=None):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))
    
        
        

        # get mean of the support
        proto = support.mean(dim=1) # Ntask x NK x d

        # add closest 10 base classes to proto and calculate mean
        # print('here ', [proto.shape, ids])
        with torch.no_grad():
            base_protos = self.get_base_protos(proto, ids)
        
        # print('base_protos', base_protos.shape)
        # print('proto ', proto.shape)
        proto = proto.reshape(proto.shape[1], 1, emb_dim)
        

        # print('proto.shape ', proto.shape)

        combined_protos = base_protos

        # print('combined_protos.shape ', combined_protos.shape)

        # print('proto.shape ', proto.shape)
        # print('combined_protos.shape ', combined_protos.shape)

    
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        
        # print('before slf attention proto, combined proto', [proto.shape, combined_protos.shape])
        
        if self.feat_attn==2:
            proto = self.self_attn2(proto, proto, proto)
        proto = self.slf_attn(proto, combined_protos, combined_protos)

        # print('after slf attention proto', [proto.shape])
        proto = proto.reshape(1, proto.shape[0], emb_dim)
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])
        # print('after slf attention proto', [proto.shape])

        if self.feat_attn==1:
            proto = self.self_attn2(proto, proto, proto)
        self.after_attn = proto

        # print('after attention ', proto.shape)
        # asd
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
                # asd
            # compute class mean
            aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
            aux_center = torch.mean(aux_emb, 2) # T x N x d
            
            if self.args.use_euclidean:
                num_query = self.args.way*(self.args.k-1)
                aux_task = aux_task.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
                aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
                # print('aux center ', aux_center.shape)
                aux_center = aux_center.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
                # print('aux center ', aux_center.shape)
                # print('aux_task ', aux_task.shape)
                aux_task_base = combined_protos.permute(1, 0, 2).reshape(-1, emb_dim).unsqueeze(1)
                # print('here ', [combined_protos.shape, aux_task_base.shape])
                
                logits_reg = - torch.sum((aux_center - aux_task_base) ** 2, 2) / self.args.temperature2
                
            else:
                aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
                aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
    
                logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1])) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)            
            
            return logits, logits_reg            
        else:
            return logits 