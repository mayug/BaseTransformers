import sys
sys.path.append('/FEAT/')
from model.models import FEAT
from model.models.feat_basetransformer2 import get_k_base
from model.utils import get_command_line_parser
from model.trainer.fsl_trainer import FSLTrainer
# from model.dataloader.mini_imagenet import MiniImageNet as Dataset
from model.dataloader.samplers import CategoriesSampler, RandomSampler, ClassSampler
from nb_utils.cache_utils import pairwise_distances_logits


import model.dataloader.mini_imagenet as mini

import os
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import pandas as pd
import numpy as np

import json

import torch.nn.functional as F
import argparse
import gc


def z_norm(features):
    d = features.shape[-1]
    features_mean = features.mean(dim=-1)
    features_std = features.std(dim=-1)
    features_znorm = (features-features_mean.unsqueeze(1).repeat(1, d))/(features_std.unsqueeze(1).repeat(1, d))
    return features_znorm


def get_dataset(args):
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'TieredImageNet_og':
        from model.dataloader.tiered_imagenet_og import tieredImageNet_og as Dataset
    print('Using dataset ', Dataset)
    return Dataset


def get_cache(model, loader, sal=None, feat=False, first_half=None):
#     model.train()
    model.eval()
    embeds_list = []
    feat_embeds_list = []
    ids_list = []
    i=0
    loader_len = len(loader)
    for data, target, id_ in tqdm(loader):
        # if first_half==True:
        #     if i <= len(loader)/2:
        #         pass
        #     else:
        #         break
        # elif first_half==False:
        #     if i > len(loader)/2:
        #         pass
        #     else:
        #         continue
        # else:
        #     pass
            
            
        if sal:
            data = sal_utils.get_saliency_crops2(data, use_minmax=False, thresh=0.6)
        with torch.no_grad():
            a = model.encoder(data.cuda())
#             print(['a ', a.shape, a.min(), a.max()])
            embeds_list.append(a.cpu())
            ids_list.append(id_)
            if feat:
                a = a.unsqueeze(0)
                feat_embeds = model.slf_attn(a,a,a).squeeze()
                feat_embeds_list.append(feat_embeds.cpu())
        # if i==10:
        #     break
        i = i + 1
    print('len of embeds ', len(embeds_list), embeds_list[0].shape)
    if feat:
        embeds = torch.cat(feat_embeds_list)
    else:
        embeds = torch.cat(embeds_list)
    ids = [id_ for ids in ids_list for id_ in ids]
    save_dict = {'embeds':embeds, 'ids':ids}
    print('embeds shape, len of ids ', [embeds.shape, len(ids)])
    return save_dict


# modify ids_cache to work with cub dataset
def get_ids_cache(query_cache, embeds_cache, remove_instances, save_top=200, random=False):
    print('Random is ', random)
    all_ids = embeds_cache['ids']
    all_classes = ids2classes(all_ids)
    all_proto = embeds_cache['embeds']
    save_dict = {}
    all_ind = np.arange(start=0, stop=len(all_ids), step=1)
    i = 0
    for id_, a in tqdm(zip(query_cache['ids'], query_cache['embeds']), total=len(query_cache['ids'])):

            with torch.no_grad():
                similarities,masks = pairwise_distances_logits_single(a.unsqueeze(0).cuda(),
                                                                      all_proto.cuda(),
                                                                      list([id_]),
                                                                      remove_instances=remove_instances,
                                                                      all_classes=all_classes)
            similarities = similarities[0].cpu()
            mask = masks[0]
            if similarities.dim() ==1:
                similarities = similarities.unsqueeze(0)
            similarities_sorted = torch.sort(similarities, descending=True, dim=1)

            a_ind = similarities_sorted.indices.squeeze()

            if random:
                masked_ind = all_ind[~mask]

                random_ind = torch.randint(low=0, high=len(masked_ind), size=(len(masked_ind),))

                top_ind = masked_ind[random_ind][:save_top]

            else:
                top_ind = np.array(all_ind[~mask][a_ind][:save_top])

            all_ids = np.array(all_ids)
            # save_dict[id_] = [all_ids[top_ind], top_ind, mask]
            save_dict[id_] = top_ind

            # if i ==10:
            #     break
            # i=i+1

    
    return save_dict


def get_ids_cache_parallel(query_cache, embeds_cache, remove_instances, save_top=100, random=False):
    print('Random is ', random)

    save_dict = {}
    results_list = []

    f_partial = partial(get_nn, query_cache=query_cache, 
        embeds_cache=embeds_cache, 
        remove_instances=remove_instances, 
        random=random,
        save_top=save_top)
    all_ids_len = len(query_cache['ids'])
    with Pool(4) as p:
        for results in tqdm(p.imap(f_partial, range(all_ids_len)), total=all_ids_len):
            results_list.append(results)
    # for i in tqdm(range(len(query_cache['ids']))):
    #     results_list.append(get_nn(i, query_cache=query_cache, 
    #         embeds_cache=embeds_cache, remove_instances=remove_instances, 
    #         random=random, save_top=save_top))
    
    for r in results_list:
        save_dict.update(r)

    return save_dict


def get_nn(i, query_cache=None, embeds_cache=None, 
    remove_instances=True,  random=False, save_top=100):
    id_, a = query_cache['ids'][i], query_cache['embeds'][i]
    all_ids = embeds_cache['ids']
    all_classes = ids2classes(all_ids)
    all_proto = embeds_cache['embeds']
    all_ind = np.arange(start=0, stop=len(all_classes), step=1)
    with torch.no_grad():
        similarities,masks = pairwise_distances_logits_single(a.unsqueeze(0).cuda(),
                                                                all_proto.cuda(),
                                                                list([id_]),
                                                                remove_instances=remove_instances,
                                                                all_classes=all_classes)
    similarities = similarities[0].cpu()
    mask = masks[0]
    if similarities.dim() ==1:
        similarities = similarities.unsqueeze(0)
    similarities_sorted = torch.sort(similarities, descending=True, dim=1)

    a_ind = similarities_sorted.indices.squeeze()

    if random:
        masked_ind = all_ind[~mask]

        random_ind = torch.randint(low=0, high=len(masked_ind), size=(len(masked_ind),))

        top_ind = masked_ind[random_ind][:save_top]

    else:
        top_ind = np.array(all_ind[~mask][a_ind][:save_top])

    all_ids = np.array(all_ids)
    return {id_: [all_ids[top_ind], top_ind, mask]}



def ids2classes(ids):
    classes = np.array([id_[:-len('00000005')] for id_ in ids])
    return classes

def pairwise_distances_logits_single(query, all_proto, ids, 
                                     remove_instances,
                                     all_classes,
                                     distance_type='euclid',
                                     use_z_norm=True):

    similarities_list = []
    masks_list = []
    current_classes = list(ids2classes(ids))
    if remove_instances:
        for query, curr in zip(query, current_classes):
            mask = np.zeros(all_classes.shape).astype(np.bool)
            fitered_ids = np.argwhere(curr==all_classes)
            mask[fitered_ids] = 1
            all_proto_masked = all_proto[~mask]
            query = query.unsqueeze(0)
            if use_z_norm:
                query = z_norm(query)
                all_proto_masked = z_norm(all_proto_masked)
            similarities = pairwise_distances_logits(query, all_proto_masked).squeeze()
            similarities_list.append(similarities)
            masks_list.append(mask)
        
    return similarities_list, masks_list

# create ids cache for convnet; to check if performance matches
def get_ids_cache_all(embeds_cache, query_cache, query_cache_val, query_cache_test, 
                      save_path, random=False):
    save_dict = get_ids_cache(query_cache, 
              embeds_cache=embeds_cache,
              remove_instances=True, random=random)
    print('save dict ', len(save_dict))

    torch.save(save_dict, f'./embeds_cache/temp/save_dict_train.pt')

    save_dict_val = get_ids_cache(query_cache_val, 
              embeds_cache=embeds_cache,
              remove_instances=True, random=random)

    print('save dict_val ', len(save_dict_val))
    torch.save(save_dict_val, f'./embeds_cache/temp/save_dict_val.pt')


    save_dict_test = get_ids_cache(query_cache_test, 
              embeds_cache=embeds_cache,
              remove_instances=True, random=random)
    
    print('save dict_test ', len(save_dict_test))
    torch.save(save_dict_test, f'./embeds_cache/temp/save_dict_test.pt')

    save_dict.update(save_dict_val)
    save_dict.update(save_dict_test)

    del save_dict_val
    del save_dict_test

    gc.collect()

    print('save dict final ', len(save_dict))
    torch.save(save_dict, save_path)
    print('saved in ', save_path)


def get_save_dict(main_args):
    print(main_args)
    parser = get_command_line_parser()

    args = parser.parse_args([])

    args.mixed_precision = None
    args.save_path = 'random/'
    args.orig_imsize = -1
    args.wandb_mode='disabled'


    args.init_weights = main_args.init_weights_path
    args.backbone_class = main_args.backbone_class
    args.dim_model=main_args.dim_model

    args.num_classes = args.way
    
    if main_args.type == '1d':
        args.resize = 1
        if main_args.backbone_class == 'ConvNet':
            args.max_pool = 'max_pool'
        else:
            args.max_pool = 'avg_pool'
    elif main_args.type == '2d':
        args.max_pool = None
        args.resize = 0
    else:
        raise NotImplementedError
    
    print(args)
    trainer = FSLTrainer(args)

    args.dataset = main_args.main_dataset
    Dataset = get_dataset(args)
    # print('here ', args.dataset)
    trainset = Dataset('train', args, return_id=True)
    # print('here ', [trainset, len(trainset)])
    # asd
    # trainset = mini.MiniImageNet('train', args, return_id=True)

    valset = Dataset('val', args, return_id=True)
    testset = Dataset('test', args, return_id=True)

    # trainset_codes = trainset.wnids

    batch_size=80

    train_loader = DataLoader(trainset, shuffle=False, batch_size=batch_size)
    val_loader = DataLoader(valset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(testset, shuffle=False, batch_size=batch_size)

    if main_args.dataset == 'train':
        loader = train_loader
    elif main_args.dataset == 'val':
        loader = val_loader
    elif main_args.dataset == 'test':
        loader = test_loader

    print('main_args.first_half ', [main_args.first_half, type(main_args.first_half)])
    save_dict = get_cache(trainer.model, loader, feat=False, first_half=main_args.first_half)

    return save_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_weights_path', type=str, default=None)
    parser.add_argument('--backbone_class', type=str, default='ConvNet')
    parser.add_argument('--dim_model', type=int, default=64)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=80)
    # parser.add_argument('--fast_query', type=bool, default=False)
    parser.add_argument('--type', type=str, default='1d') # can be 1d, 2d or fastq for fast_cache
    parser.add_argument('--dataset', type=str, default='train')# train, test, val

    parser.add_argument('--main_dataset', type=str, default='MiniImagenet')
    parser.add_argument('--first_half', type=str, default=None)

    parser.add_argument('--train_cache', type=str, default=None)
    parser.add_argument('--val_cache', type=str, default=None)
    parser.add_argument('--test_cache', type=str, default=None)
    parser.add_argument('--base_cache', type=str, default=None)
    
    main_args = parser.parse_args()

    main_args.first_half = bool(main_args.first_half)


    assert main_args.type in main_args.save_path

    if main_args.type == '1d' or main_args.type == '2d':
        save_dict = get_save_dict(main_args)
        print('save_dict ', [save_dict['embeds'].shape, save_dict['embeds'].min(), save_dict['embeds'].max()])
        torch.save(save_dict, main_args.save_path)
        print('saved in ', main_args.save_path)

    elif main_args.type == 'fastq':
        if main_args.train_cache is not None:
            embeds_cache = torch.load(main_args.base_cache)
            query_cache = torch.load(main_args.train_cache)
            val_cache = torch.load(main_args.val_cache)
            test_cache = torch.load(main_args.test_cache)

        else:
            print('Creating train, val and test cache from scratch and saving in embeds_cache/temp/')
            main_args.type='1d'
            for dataset in ['train', 'val', 'test']:
                main_args.dataset = dataset
                save_dict = get_save_dict(main_args)
                print('save_dict ', [save_dict['embeds'].shape, save_dict['embeds'].min(), save_dict['embeds'].max()])
                torch.save(save_dict, f'./embeds_cache/temp/{dataset}.pt')
                print('saved in embeds_cache/temp/', dataset)
            
            save_dict = None

            embeds_cache = torch.load('./embeds_cache/temp/train.pt')
            print('WARNING!!! Using query train cache as base_cache, BE CAREFUL')
            query_cache = embeds_cache
            val_cache = torch.load('./embeds_cache/temp/val.pt')
            test_cache = torch.load('./embeds_cache/temp/test.pt')

        save_path = main_args.save_path
        get_ids_cache_all(embeds_cache, query_cache, val_cache, test_cache, save_path)


    else:
        raise NotImplementedError        
 




