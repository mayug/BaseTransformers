import pandas as pd
import json
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os


class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []
cwd = os.getcwd()
print('here ', cwd)
# asd
wnids = json.load(open('../notebooks/wnids.json', 'r'))

trainset_codes = wnids['train']
valset_codes = wnids['val']
testset_codes = wnids['test']

def get_class(number, dataset='test'):
    class_names = pd.read_csv('./data/miniimagenet/split/class_names.csv', header=None, delimiter=' ')
    class_names.columns = ['code', 'name']
    

    
    if dataset=='train':
        return class_names[class_names['code']==trainset_codes[number]]['name'].item()
    elif dataset=='val':
        return class_names[class_names['code']==valset_codes[number]]['name'].item()
    elif dataset=='test':
        return class_names[class_names['code']==testset_codes[number]]['name'].item()

def get_class_from_code(code):
    class_names = pd.read_csv('./data/miniimagenet/split/class_names.csv', header=None, delimiter=' ')
    class_names.columns = ['code', 'name']
    return class_names[class_names['code']==code]['name'].item()
    

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
    
    

# proto_dict = pickle.load(open('../notebooks/proto_dict_new.pkl', 'rb'))  
# all_proto = torch.cat([torch.Tensor(proto_dict[i]).unsqueeze(0) for i in range(len(proto_dict))], dim=0).cuda()


# def get_k_base(proto, all_proto, return_weights=False, k=10):
#     proto = torch.Tensor(proto)
# #     print('proto ', proto.shape)
#     # print('using k={}'.format(str(k)))
#     similarities = pairwise_distances_logits(proto, all_proto).squeeze()
# #     print('sim', [similarities.shape, similarities.dim()])
#     if similarities.dim() ==1:
#         similarities = similarities.unsqueeze(0)
#     similarities_sorted = torch.sort(similarities, descending=True, dim=1)
#     a = similarities_sorted.values
#     a = F.softmax(a[:,1:], dim=1)
#     a_ind = similarities_sorted.indices
#     if return_weights:
#         return a_ind[:, 1:k], a[:, :k-1]
#     return a_ind[:, 1:k]


# given a class get k nearest samples of images from class
def get_k_images(proto, trainer, class_number=None, k=10):
    start_time = time.time()
    batch_size=128
    if class_number:
        trainset
    train_loader_large = DataLoader(trainset, batch_size=batch_size)
    i = 0
    model = trainer.model.eval()
    proto = torch.Tensor(proto_dict[10]).cuda().unsqueeze(0)
    store_logits = []
    for data, targets in train_loader_large:
        with torch.no_grad():
            data = data.cuda()
            embeds = model.encoder(data)
            logits = pairwise_distances_logits(embeds, proto).squeeze().cpu().numpy()
            store_logits.append(logits)
            i = i + batch_size
    end_time = time.time()
    all_sim = torch.Tensor(store_logits).reshape(-1)
    sorted_sim = torch.sort(all_sim, descending=True)
    a = torch.cat([trainset[i][0].unsqueeze(0) for i in sorted_sim.indices[:k]], dim=0)

    return a 

# plot the image given a class_name or class_number
def plot_class(class_, dataset, k=1):
    if isinstance(class_, str):
        class_number = get_class_number(class_name)
    else:
        class_number = class_
    
    if isinstance(dataset, str):
        if dataset=='train':
            dataset = trainset
        elif dataset=='val':
            dataset=valset
        elif dataset=='test':
            dataset=testset
    
    label = np.array(dataset.label)
    ids = np.argwhere(np.array(label)==class_number).squeeze()
    sel_ids = ids[np.random.randint(low=0, high=len(ids), size=k)]
    images = np.concatenate([dataset[s][0].unsqueeze(0).numpy() for s in sel_ids])
    for i in range(k):
        plot_img(i, images)
    return images

def plot_img(i, data, title=None, minmax=True):
    img = data[i,:,:,:]
    img= np.swapaxes(img, 0, 2)
    img = np.swapaxes(img,0,1)
    if minmax:
        img = (img- img.min())/(img.max()-img.min())
    print(i)
    plt.figure()
    plt.title(str(title))
    plt.imshow(img[:,])
    plt.show()

def ids2classes(ids):
    classes = np.array([id_[:-len('00000005')] for id_ in ids])
    return classes

# read_dict = torch.load('../notebooks/embeds_cache.pt')


# def plot_top_k(embedding, current_class, dataset):
#     if dataset.num_class == 64:
#         train=True
#         set_name = 'train'
#     else:
#         train=False
#         set_name = 'test'
#     embeds = read_dict['embeds']
#     all_ids = np.array(read_dict['ids'])
#     all_classes = ids2classes(all_ids)
#     print('embedding', [embedding.min(), embedding.max(), embedding.mean()])
#     top_k, mask = get_k_base(torch.Tensor(embedding).unsqueeze(0), embeds, 
#                        train=train, remove_instances=True, all_classes=all_classes,
#                       current_classes= current_class)
#     top_k = (top_k[0]).numpy()
#     print('top_k ', top_k)
# #     print('argwehere mask ', np.argwhere(mask))
#     ids_masked = all_ids[~mask]
#     print('ids_masked ', ids_masked.shape)
#     top_k_ids = ids_masked[top_k]
    
#     for id_ in top_k_ids:
#         img,t,_ = dataset.get_image(id_)
#         print('id_', id_)
#         plot_img(0, img.unsqueeze(0), title= get_class(t, set_name))

# receptive field of cnn4
def get_receptive_field(rl=5, s=[2,2,2,2], k=[3,3,3,3]):
    
    ri = rl
    for si, ki in zip(s, k):
        ri_ = si*ri + ki-si
        ri = ri_
        print(ri)

def plot_img_new(i, data, title=None, minmax=True, fig=None, ax=None, return_ax=False, alpha=1, cmap=None):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    img = data[i,:,:,:]
    img= np.swapaxes(img, 0, 2)
    img = np.swapaxes(img,0,1)
    if minmax:
        img = (img- img.min())/(img.max()-img.min())
#     print(i)
#     plt.figure()
    if ax is None or fig is None:
        fig, ax = plt.subplots()
    ax.set_title(str(title))
#     print(img[:,].shape, type(img[:,]))
    ax.imshow(img[:,], alpha=alpha, cmap=cmap)
    if return_ax:
        return fig, ax
    else:
        plt.show()

def plot_from_dataset(i, dataset):
    data,_,id_ = dataset[i]
    plot_img(0, data.unsqueeze(0))
    print('id is ', id_)
    
