
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
        
        print([query.shape, proto.shape])
        logits = torch.matmul(query, proto.transpose(1,0))
#         print('logits.shape', logits.shape)
        return logits
    