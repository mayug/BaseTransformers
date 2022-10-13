from .feat_basetransformer3 import FEATBaseTransformer3
from .feat_basetransformer2 import get_k_base, MultiHeadAttention
import torch
import numpy as np
import torch.nn.functional as F
import os
import os.path as osp
import torch.nn as nn
from tqdm import tqdm



def apply_z_norm(features):                                                                                                                                                    
    d = features.shape[-1]                                                                                                                                                     
    features_mean = features.mean(dim=-1)                                                                                                                                      
    features_std = features.std(dim=-1)                                                                                                                                        
    # print([features_mean.mean(), features_std.mean()])                                                                                                                       
    features_znorm = (features-features_mean.unsqueeze(1).repeat(1, d))/(features_std.unsqueeze(1).repeat(1, d))                                                               
    return features_znorm  



def get_choose(n_key, n_classes=5):

    choose = torch.diag(torch.ones(n_key))
    
    # now remove all the 5th elements on either side of the diagonal 
    # except for the top right quadrant diagonal and bottom left diagonal
    # because these are the distances (ai-bi) which are the only positives in each row
    
    positive_choose = torch.zeros(n_key, n_key)

    indices = torch.arange(0,n_key,n_classes)
#     print('indices ', indices)
    n_half = int(indices.shape[0]/2)
#     print('n_half', [n_half, indices[n_half]])
    indices_selected = torch.cat([indices[0:n_half], indices[n_half+1:]])
#     print(indices_selected, 'indices_selected')
    positive_index = indices[n_half]
    positive_choose_0 = torch.zeros(n_key)
    positive_choose_0[positive_index] = 1
    choose[0, indices_selected] = 1
    choose_0 = choose[0,:]
    choose_list = []
    positive_choose_list = []
    label_list = []
    for i in range(n_key):
#         print(['i', 'positive', i, torch.argmax(positive_choose_0)])
#         print('choose_0 ', choose_0)
        choose_list.append(choose_0.unsqueeze(0))
        label_list.append(torch.argmax(positive_choose_0).item())
        positive_choose_list.append(positive_choose_0.unsqueeze(0))

        choose_0 = torch.roll(choose_0, 1, dims=0)
        positive_choose_0 = torch.roll(positive_choose_0, 1, dims=0)

#     print(label_list)
    choose = torch.cat(choose_list)
    positive_choose = torch.cat(positive_choose_list)
    label = torch.Tensor(label_list)
    return choose, positive_choose





    
def load_numpy(name):
    return np.load(os.path.join('/home/mayug/projects/few_shot/notebooks/temp/', name+'.npy'))



class FEATBaseTransformer3_2d(FEATBaseTransformer3):
    def __init__(self, args):
        args.max_pool = False
        args.resize = False
        super().__init__(args)
        # these 2d embeddings of base instances are used for combination
         
        if args.embeds_cache_2d is not None:
            print('loading 2d embeds_cache from args ', args.embeds_cache_2d)
            proto_dict_2d = torch.load(args.embeds_cache_2d)
        else:
           raise ValueError('Provide embeds_cache_2d argument')

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
 
        self.spatial_dim = None # will be initialized in forward
        print('Using embed_pool type = ', self.embed_pool)
        self.label_aux = None

        self.choose = None
        self.reshape_dim = None

        self.baseinstance_2d_norm = None
        if args.baseinstance_2d_norm:
            self.baseinstance_2d_norm = nn.BatchNorm2d(self.hdim)
        
        if args.channel_reduction:
            self.channel_reduction = nn.Sequential(
                    nn.Conv2d(self.hdim, args.channel_reduction, kernel_size=1), 
                    nn.ReLU())

            self.hdim = args.channel_reduction
            self.slf_attn = MultiHeadAttention(args.n_heads, self.hdim, self.hdim, self.hdim, dropout=0.5)
        else:
            self.channel_reduction = None

    def get_simclr_logits(self, simclr_features, temperature_simclr, fc_simclr=None, max_pool=False, version=None):
        
        n_batch, n_views, n_c, spatial, _ = simclr_features.shape

        if fc_simclr is not None:
            simclr_features = fc_simclr(simclr_features)

        max_pool=True
        if max_pool:
            simclr_features = simclr_features.reshape(n_batch*n_views, n_c, spatial, spatial)
            simclr_features = F.max_pool2d(simclr_features, kernel_size=5)
            simclr_features = simclr_features.reshape(n_batch, n_views, n_c, 1, 1)

        simclr_features = simclr_features.reshape(n_batch, n_views, -1)

        # now calculate logits using euclidian loss or cosine loss;

        a = torch.cat([simclr_features[:, 0, :], simclr_features[:, 1, :]], dim=0)
        b = torch.cat([simclr_features[:, 0, :], simclr_features[:, 1, :]], dim=0)


        n_key, emb_dim = a.shape
        n_query = b.shape[0]
        a = a.unsqueeze(0).expand(n_query, n_key, emb_dim)
        b = b.unsqueeze(1)
        logits_simclr = - torch.mean((a - b) ** 2, 2) / temperature_simclr

        # remove diagonal elements

        if version=='ver2.2':
            n_classes = 5
            if self.label_aux is None:
                choose, positive_choose = get_choose(n_key, n_classes)
                self.reshape_dim = int(choose.sum(1)[0].item())
                choose = ~(choose.bool())
                label_aux = positive_choose[choose].reshape(n_query, n_key-self.reshape_dim).argmax(1)
                self.label_aux = label_aux.cuda()
                self.choose = choose

            logits_simclr = logits_simclr[self.choose].reshape(n_query, n_key-self.reshape_dim)

        else:
            choose = ~torch.diag(torch.ones(n_key, dtype=torch.bool))
            logits_simclr = logits_simclr[choose].reshape(n_query, n_key-1)


        return logits_simclr


    def save_as_numpy(self, tensor, name, debug=False):
        if self.debug:
            if isinstance(tensor, torch.Tensor):
                np_array = tensor.detach().cpu().numpy()
            else:
                np_array = tensor
            np.save(os.path.join('/home/mayug/projects/few_shot/notebooks/temp/', name+'.npy'), np_array)
    def convert_index_space(self, id_):
        masked_ids = self.proto_dict_2d['masked_ids']

        return np.argwhere(masked_ids==id_).item()
    def get_base_protos(self, proto, ids):
        # 2d version of get_base_protos
        # querying uses 1d proto
        # but returns 3d feature maps of top_k protos

        proto = proto.squeeze()
        all_proto = self.all_proto

        if self.args.backbone_class == 'ConvNet':
            proto = F.max_pool2d(proto, kernel_size=5).squeeze()
        else:
            proto = F.adaptive_avg_pool2d(proto, output_size=(1,1)).squeeze()

        if ids is not None:
            current_classes = list(set(self.ids2classes(ids)))
            remove_instances = True
        else:
            current_classes = []
            remove_instances = False

        if self.args.query_model_path is not None:
            proto = self.get_proto_new(ids[:5])
            all_proto = self.all_proto_new

        top_k, mask = get_k_base(proto, all_proto, k=self.args.k, remove_instances=remove_instances, 
                           all_classes=self.all_classes,
                           current_classes=current_classes,
                           train=self.training,
                           random=self.random)
        self.top_k = (top_k, mask)
    
        self.save_as_numpy(top_k, 'top_k')
        self.save_as_numpy(mask, 'mask')

        all_proto = self.all_proto_2d[~mask]


        base_protos = all_proto[top_k, :]


        return base_protos

    def get_base_protos_fast_query(self, ids, k=None):
        # this code is wrong, need to have different code for train and test time
        # Also remove_instances of same class    
        # Update: code is correct, it's done during the cache creation itself

        # check which of the following two statements take more time and optimize
        if k is None:
            k=self.args.k
        
  
        top_indices = np.stack([self.query_dict[id_][1][:k] for id_ in ids], axis=0)
        base_protos = self.all_proto_2d[torch.Tensor(top_indices).long()]
        
        return base_protos


    def _forward(self, instance_embs, support_idx, query_idx, ids=None, simclr_embs=None):
        
        if self.channel_reduction:
            instance_embs = self.channel_reduction(instance_embs)

        spatial_dim = instance_embs.shape[-1]
        self.spatial_dim = spatial_dim
        self.save_as_numpy(instance_embs, 'instance_embs')

        emb_dim = instance_embs.size(-3)
        num_patches = np.prod(instance_embs.shape[-2:]) 

        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (emb_dim, spatial_dim, spatial_dim,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (emb_dim, spatial_dim, spatial_dim,)))
    
        self.save_as_numpy(support, 'support')
        self.save_as_numpy(support, 'query')

        proto = support.mean(dim=1) 
        self.save_as_numpy(proto, 'proto')
        n_class = proto.shape[1]
        n_batch = proto.shape[0]
        k = self.args.k

        # get topk base instances
 
        if self.fast_query is None:
            # with torch.no_grad():
            base_protos = self.get_base_protos(proto, ids) # 
        else:
            base_protos = self.get_base_protos_fast_query(ids[:5])
            # base_protos = self.get_base_protos_fast_query_ti_temp(ids[:5])
        self.save_as_numpy(base_protos, 'base_protos')
        # base_protos n_class * topk * 64 * (5 * 5)

        if self.baseinstance_2d_norm:
            base_protos = base_protos.reshape(n_class*k, emb_dim, spatial_dim, spatial_dim)
            base_protos = self.baseinstance_2d_norm(base_protos)
            base_protos = base_protos.reshape(n_class, k, emb_dim, spatial_dim, spatial_dim)


        if self.channel_reduction:
            base_k = base_protos.shape[1]
            base_emb_dim = base_protos.shape[2]
            base_protos = base_protos.reshape(n_class*base_k, base_emb_dim, spatial_dim, spatial_dim)
            base_protos = self.channel_reduction(base_protos)
            base_protos = base_protos.reshape(n_class, base_k, emb_dim, spatial_dim, spatial_dim)            


        if self.args.z_norm=='before_tx' or self.args.z_norm=='both':                                                                                                          
                                                                                                                                                                               
            b1, b2, b3, b4, _ = base_protos.shape                                                                                                                              
            p1, p2, p3, p4, _ = proto.shape                                                                                                                                    
            base_protos = base_protos.reshape(b1*b2, b3*b4*b4)                                                                                                                 
            proto = proto.reshape(p1*p2, p3*p4*p4)                                                                                                                             
                                                                                                                          
            base_protos, proto = apply_z_norm(base_protos), apply_z_norm(proto)                                                                                                
            base_protos = base_protos.reshape(b1,b2,b3,b4,b4)                                                                                                                  
            proto = proto.reshape(p1,p2,p3,p4,p4)                                                                                                                              



        proto = proto.reshape(proto.shape[1], emb_dim, -1).permute(0, 2, 1).contiguous()
        self.save_as_numpy(proto, 'proto_after_reshape')

        base_protos = base_protos.permute(0, 2, 1, 3, 4).contiguous()
        combined_protos = base_protos.reshape(n_class*n_batch, emb_dim, -1).permute(0, 2, 1).contiguous()
        self.save_as_numpy(combined_protos, 'combined_protos')

    
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        

        if self.feat_attn==2:
            proto = self.self_attn2(proto, proto, proto)
        proto = self.slf_attn(proto, combined_protos, combined_protos)

        proto = proto.permute(0, 2, 1).contiguous()
       
        
        proto = proto.view(-1, emb_dim, spatial_dim, spatial_dim)

        query = query.view(-1, emb_dim, spatial_dim, spatial_dim)


        if isinstance(self.embed_pool, torch.nn.Identity):
            # print(proto.shape)
            # print(query.shape)
            proto = proto.reshape(proto.shape[0], -1).unsqueeze(0)
            query = query.reshape(query.shape[0], -1)

            emb_dim = emb_dim*(spatial_dim**2)

        else:
            proto = self.embed_pool(proto, kernel_size=5).squeeze().unsqueeze(0)
            query = self.embed_pool(query, kernel_size=5).squeeze()
       
            
  

        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        if self.feat_attn==1:
            proto = self.self_attn2(proto, proto, proto)

        self.after_attn = proto

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
                # implementing simclr loss on the encoder embeddings
            if  simclr_embs is not None:

                if self.args.simclr_loss_type=='ver1':
                    sim_k = simclr_embs[:,0,:,:,:]
                    sim_q = simclr_embs[:,1,:,:,:]
                    if self.args.backbone_class == 'ConvNet':
                        sim_k = F.max_pool2d(sim_k, kernel_size=self.spatial_dim)
                        sim_q = F.max_pool2d(sim_q, kernel_size=self.spatial_dim)
                    n_simclr_eps = self.args.shot + self.args.query
                    n_proto = self.args.way
                    n_q = self.args.way
                    sim_k = sim_k.reshape(n_simclr_eps, n_proto, -1)
                    sim_q = sim_q.reshape(n_simclr_eps, n_proto, -1)
                    sim_clr_emb_dim = sim_q.shape[-1]
                    
                    sim_k = sim_k.unsqueeze(1).expand(n_simclr_eps, n_q, n_proto, sim_clr_emb_dim)
                    sim_q = sim_q.unsqueeze(2)

                    logits_simclr = - torch.mean((sim_k - sim_q) ** 2, 3) / self.args.temperature2

                elif self.args.simclr_loss_type=='ver2.1' or self.args.simclr_loss_type=='ver2.2':
                    fc_simclr = None

                    logits_simclr = self.get_simclr_logits(simclr_embs,
                        temperature_simclr=self.args.temperature2,
                        fc_simclr=fc_simclr,
                        version=self.args.simclr_loss_type) 
                return logits, logits_simclr
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
                aux_emb = self.slf_attn(aux_task, combined_protos, combined_protos)
            # compute class mean
            aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
            aux_center = torch.mean(aux_emb, 2) # T x N x d
            
            if self.args.use_euclidean:
                aux_task = aux_task.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
                aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
                aux_center = aux_center.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

                logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
            else:
                aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
                aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
    
                logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1]).contiguous) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)            
            
            return logits, logits_reg            
        else:
            return logits
    
    def update_2d_embeds(self, loader):
        self.eval()
        embeds_list = []
        ids_list = []
        i=0
        loader_len = len(loader)
        for data, target, id_ in tqdm(loader):
            data = data.cuda()
            if self.args.mixed_precision is not None:
                data = data.half()
            with torch.no_grad():

                a = self.encoder(data)
                embeds_list.append(a.cpu())
                ids_list.append(id_)
            i = i + 1
        embeds = torch.cat(embeds_list)
        ids = [id_ for ids in ids_list for id_ in ids]
        proto_dict_2d = {'embeds':embeds, 'ids':ids}
        self.proto_dict_2d = proto_dict_2d
        self.all_proto_2d = proto_dict_2d['embeds'].cuda()
        if self.args.mixed_precision is not None and self.args.mixed_precision!='O0':
            self.all_proto_2d = self.all_proto_2d.half()
        self.train()     


def normalize(x, dim=-1, p=2):
    norm = x.norm(p=p, dim=dim, keepdim=True)
    x_normalized = x.div(norm)
    return x_normalized