import os
import shutil
import time
import pprint
import torch
import argparse
import numpy as np

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()    
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)

    return encoded_indicies

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def ensure_path(dir_path, scripts_to_save=None):
    if os.path.exists(dir_path):
        if input('{} exists, remove? ([y]/n)'.format(dir_path)) != 'n':
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for src_file in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(src_file))
            print('copy {} to {}'.format(src_file, dst_file))
            if os.path.isdir(src_file):
                shutil.copytree(src_file, dst_file)
            else:
                shutil.copyfile(src_file, dst_file)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


# function to calculate class accuracies
# some way of getting class names in each iteration 

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
#     print('pred ', pred)
#     print('label ', label)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


class AccuracyClassAverager():
    def __init__(self, n_classes):
        self.class_acc_dict = {i:0 for i in range(n_classes)}
        self.n_dict = {i:0 for i in range(n_classes)}
    
    
    def add(self, logits, label):
        way = logits.shape[1]
        classes = label[:way]
        # print('classes ', classes)
        for c in classes:
            c = c.item()
#             print('class ', c)
#             print('label==c ', label==c)
            label_c = torch.ones((label==c).sum())*np.argwhere((classes==c).int().cpu())
            logits_c = logits[label==c, :]
#             print('here')
            
#             print(label_c)
#             print(logits_c)
#             print(np.argwhere(label_c))
#             print('label_c', label_c)
#             print('logits_c', logits_c)
            
            x = count_acc(logits_c, label_c)
#             print(x)
            self.class_acc_dict[c] = (self.class_acc_dict[c] * self.n_dict[c] + x) / (self.n_dict[c] + 1)
            self.n_dict[c]  = self.n_dict[c] + 1
    
    def item(self):
        return self.class_acc_dict

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    # print('a,b', [a.shape, b.shape])
    logits = -((a - b)**2).sum(dim=2)
    # print('logits', [logits.shape])
    return logits

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def postprocess_args(args):            
    args.num_classes = args.way
    save_path1 = '-'.join([args.dataset, args.model_class, args.backbone_class, '{:02d}w{:02d}s{:02}q'.format(args.way, args.shot, args.query)])
    save_path2 = '_'.join([str('_'.join(args.step_size.split(','))), str(args.gamma),
                           'lr{:.2g}mul{:.2g}'.format(args.lr, args.lr_mul),
                           str(args.lr_scheduler), 
                           'T1{}T2{}'.format(args.temperature, args.temperature2),
                           'b{}'.format(args.balance),
                           'bsz{:03d}'.format( max(args.way, args.num_classes)*(args.shot+args.query) )
                           ])    
    if args.init_weights is not None:
        save_path1 += '-Pre'
    if args.use_euclidean:
        save_path1 += '-DIS'
    else:
        save_path1 += '-SIM'

    if args.k:
        save_path2 += '_k{}'.format(str(args.k))
    if args.base_wt:
        save_path2 += '_bwt{}'.format(str(args.base_wt))

    save_path2 += '_{}'.format(str(time.strftime('%Y%m%d_%H%M%S')))

    if args.fix_BN:
        save_path2 += '-FBN'
    if not args.augment:
        save_path2 += '-NoAug'

    
        
    
            
    if not os.path.exists(os.path.join(args.save_dir, save_path1)):
        os.mkdir(os.path.join(args.save_dir, save_path1))
    args.save_path = os.path.join(args.save_dir, save_path1, save_path2)
    return args

def get_command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=600)
    parser.add_argument('--model_class', type=str, default='FEAT', 
                        choices=['MatchNet', 'ProtoNet', 'BILSTM', 'DeepSet', 'GCN', 'FEAT', 'FEATSTAR', 
                        'FEATv3_1', 'FEATv3_2_2', 'SemiFEAT', 'SemiProtoFEAT',
                        'FEATBaseClass', 'FEATBaseTransformer', 'FEATSAL', 'FEATBaseTransformer2', 
                        'FEATBaseClass2', 'FEATBaseSalCrop', 'FEATSAL2', 'FEATBaseTransformer3',
                        'FEAT_Aux', 'FEATBaseTransformer3_Aux', 'BIT3_pretrain', 'FEATBaseTransformer3_2d',
                        'FEATBaseTransformer3_2d_sym', 'FEATBaseTransformer3_2d_patch',
                        'FEATBaseTransformer3_2d_ctx', 'FEATBaseTransformer3_2d_5shot',
                        'FEATBaseTransformer3_2d_5shot_v2',
                        'FEATBaseTransformer3_2d_5shot_v3',
                        'FEATBaseTransformer3_2d_sym_ctx',
                        'FEATBaseTransformer3_2d_ctx_mix']) # None for MatchNet or ProtoNet
    parser.add_argument('--base_wt', type=float, default=None)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--update_base_interval', type=int, default=None)
    parser.add_argument('--update_base_embeds', type=bool, default=False)
    parser.add_argument('--patience', type=int, default=10) # number of epochs after which base_embeds update will run
    parser.add_argument('--base_protos', type=int, default=1)

    parser.add_argument('--feat_attn', type=int, default=0) # this is only for feat_basetransformer2
    parser.add_argument('--query_attn', type=int, default=0)

    parser.add_argument('--sal_crop', type=str, default=None)
    parser.add_argument('--random', type=int, default=0) # just use random base instances instead of top_k, k is the number of random base instances

    parser.add_argument('--max_pool', type=str, default='max_pool')
    parser.add_argument('--resize', type=int, default=1)

    parser.add_argument('--use_euclidean', action='store_true', default=False)    
    parser.add_argument('--backbone_class', type=str, default='ConvNet',
                        choices=['ConvNet', 'Res12', 'Res18', 'WRN', 'Res12_ptcv', 'Res12_info', 'WRN_S2M2'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'TieredImageNet_og', 'CUB'])
    
    parser.add_argument('--n_heads', type=int, default=1) # number of heads in baseinstancetransformer3

    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--eval_shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--eval_query', type=int, default=15)
    parser.add_argument('--balance', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--temperature2', type=float, default=1)  # the temperature in the  
     
    # optimization parameters
    parser.add_argument('--orig_imsize', type=int, default=-1) # -1 for no cache, and -2 for no resize, only for MiniImageNet and CUB
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_mul', type=float, default=10)    
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine', 'onecycle', 'cyclic'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.2)    
    parser.add_argument('--fix_BN', action='store_true', default=False)     # means we do not update the running mean/var in BN, not to freeze BN
    parser.add_argument('--fix_BN_only', action='store_true', default=False)     # same as above but eval only applied to BN layers

    parser.add_argument('--augment',   action='store_true', default=False)
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--init_weights', type=str, default=None) # initialize the BaseTransformer
    parser.add_argument('--init_weights_tx', type=str, default=None)
    
    parser.add_argument('--mixed_precision', type=str, default=None) # for old non amp checkpoints
    # default functionality is None

    # usually untouched parameters
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005) # we find this weight decay value works the best
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')


    # test checkpoint
    parser.add_argument('--test', type=str, default=None)
    
    # debug
    parser.add_argument('--debug', type=int, default=0)

    # top_k querying using a different model;
    parser.add_argument('--query_model_path', type=str, default=None)
    parser.add_argument('--pca_dim', type=int, default=None) # dimension for pca while querying

    parser.add_argument('--pass_ids', type=int, default=0) # this is for instancetransformers; to prevent same class instances in topk
    parser.add_argument('--remove_instances', type=int, default=0)
    # be careful; both pass_ids and remove_instances should be 1 for old functionality of pass_ids=1

    parser.add_argument('--image_size', type=int, default=None)

    # this is for proto_bit3_2d; whats pooling before final euclidian calc;
    parser.add_argument('--embed_pool', type=str, default='post_loss_avg')

    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--drop_rate', type=float, default=0.1)

    parser.add_argument('--dim_model', type=int, default=64)

    parser.add_argument('--tx_k_v', type=int, default=None)

    # fast_query
    parser.add_argument('--fast_query', type=str, default=None)

    # embeds_cache 1d and 2d
    parser.add_argument('--embeds_cache_1d', type=str, default=None)
    parser.add_argument('--embeds_cache_2d', type=str, default=None)
    parser.add_argument('--baseinstance_2d_norm', type=str, default=None)

    parser.add_argument('--channel_reduction', type=int, default=None)


    parser.add_argument('--wandb_mode', type=str, default='disabled')

    parser.add_argument('--n_patches', type=int, default=None)

    # parser.add_argument('--cockpit', type=str, default=None)

    parser.add_argument('--return_simclr', type=int, default=None) # number of views in simclr
    parser.add_argument('--label_aux_type', type=str, default=None) # number of views in simclr

    parser.add_argument('--simclr_loss_type', type=str, default=None)

    parser.add_argument('--config', type=str, default=None)

    parser.add_argument('--base_protos_path', type=str, default=None)

    # ctx models
    parser.add_argument('--attn_model', type=str, default=None)
    parser.add_argument('--do_query_attn', type=int, default=1)

    parser.add_argument('--z_norm', type=str, default=None, choices=['before_tx', 'before_euclidian', 
                                                                    'both', None])

    return parser
