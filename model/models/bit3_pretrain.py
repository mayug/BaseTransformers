from .feat_basetransformer3 import FEATBaseTransformer3
import torch
import numpy as np
import torch.nn.functional as F


# BaseTransformer3 --> folds the prorotypes into batch dimension
# so attention between 1 prototype and its topk base instances

class BIT3_pretrain(FEATBaseTransformer3):
    def _forward(self, instance_embs, support_idx=None,
    query_idx=None, ids=None):
        emb_dim = instance_embs.size(-1)

        # print('instance_embds', instance_embs.shape)
        # print('ids ', len(ids))
        # add closest 10 base classes to proto and calculate mean
        # print('here ', [instance_embs.shape, ids])
        base_protos = self.get_base_protos(instance_embs, ids)
        

        # Now pass through transformer and get the reconstructed protos for each class
        proto = instance_embs.unsqueeze(1) 
        # print('base_protos', [proto.shape, base_protos.shape])

        proto = self.slf_attn(proto, base_protos, base_protos)
        # print('after slf proto ', proto.shape)  
        proto = proto.squeeze()
        if self.training:
            return proto, None
        return proto
        asd