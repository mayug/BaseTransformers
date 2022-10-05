import argparse
import os
import os.path as osp
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.models.classifier import Classifier
from model.dataloader.samplers import CategoriesSampler
from model.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric

from model.utils import get_command_line_parser
from model.trainer.fsl_trainer import FSLTrainer
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np

# pre-train model, compute validation acc after 500 epoches


def get_model():
    parser = get_command_line_parser()

    args = parser.parse_args([])


    args.save_path = 'random/'
    args.num_classes = args.way
    args.init_weights = '/home/mayug/projects/few_shot/FEAT/saves/initialization/miniimagenet/con-pre.pth'
    args.model_class = 'BIT3_pretrain'
    # args.gpu=0
    args.num_classes=5
    args.way=5
    args.shot=1
    args.query=15

    args.eval_way=5
    args.eval_shot=1
    args.eval_query=15

    args.use_euclidean = True
    args.temperature = 64
    args.temperature2 = 16

    args.k = 50
    args.base_protos = 0
    args.feat_attn = 0
    args.pass_ids = 1 

    trainer = FSLTrainer(args)
    return trainer.model 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'TieredImagenet', 'CUB'])    
    parser.add_argument('--backbone_class', type=str, default='ConvNet', choices=['ConvNet', 'Res12'])
    parser.add_argument('--schedule', type=str, default="75, 150, 300", help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--query', type=int, default=15)    
    parser.add_argument('--resume', type=bool, default=False)
    args = parser.parse_args()
    args.orig_imsize = 128
    pprint(vars(args))
    
    args.schedule = [int(i) for i in args.schedule.split(',')]
    print('schedule ', args.schedule)
    # asd
    save_path1 = '-'.join([args.dataset, args.backbone_class, 'Pre-random'])
    save_path2 = '_'.join([str(args.lr), str(args.gamma), str(args.schedule)])
    args.save_path = osp.join(save_path1, save_path2)
    if not osp.exists(save_path1):
        os.mkdir(save_path1)
    ensure_path(args.save_path)
    print('save path for this is ', args.save_path)
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImagenet':
        from model.dataloader.tiered_imagenet import tieredImageNet as Dataset    
    else:
        raise ValueError('Non-supported Dataset.')

    trainset = Dataset('train', args, augment=True, return_id=True)
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    args.num_class = trainset.num_class
    valset = Dataset('val', args)
   # test on 16-way 1-shot

    shot = 1
    query = 15
    way = 5
    categories_sampler = CategoriesSampler(trainset.label,
                                    1000,
                                    way,
                                    shot+query)
    val_sampler = CategoriesSampler(valset.label, 200, way,shot+query)
    train_loader_categories = DataLoader(trainset, batch_sampler=categories_sampler)

    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    args.way = valset.num_class
    args.shot = 1
    
    # construct model
    model = get_model()

    # optimizer taking only attention parameters
    optimizer = torch.optim.Adam(model.slf_attn.parameters(), lr=args.lr, weight_decay=0.0005)

    criterion = torch.nn.MSELoss()
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if args.ngpu  > 1:
            model.encoder = torch.nn.DataParallel(model.encoder, device_ids=list(range(args.ngpu)))
        
        model = model.cuda()
        criterion = criterion.cuda()
    
    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
    
    def save_checkpoint(is_best, filename='checkpoint.pth.tar'):
        state = {'epoch': epoch + 1,
                 'args': args,
                 'state_dict': model.state_dict(),
                 'trlog': trlog,
                 'val_loss': trlog['val_loss'],
                 'optimizer' : optimizer.state_dict(),
                 'global_count': global_count}
        
        torch.save(state, osp.join(args.save_path, filename))
        if is_best:
            shutil.copyfile(osp.join(args.save_path, filename), osp.join(args.save_path, 'model_best.pth.tar'))
    
    if args.resume == True:
        # load checkpoint
        state = torch.load(osp.join(args.save_path, 'model_best.pth.tar'))
        init_epoch = state['epoch']
        resumed_state = state['state_dict']
        # resumed_state = {'module.'+k:v for k,v in resumed_state.items()}
        model.load_state_dict(resumed_state)
        trlog = state['trlog']
        optimizer.load_state_dict(state['optimizer'])
        initial_lr = optimizer.param_groups[0]['lr']
        global_count = state['global_count']
    else:
        init_epoch = 1
        trlog = {}
        trlog['args'] = vars(args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['min_loss'] = 1e10
        trlog['min_loss_epoch'] = 0
        
        initial_lr = args.lr
        global_count = 0

    timer = Timer()
    writer = SummaryWriter(logdir=args.save_path)
    for epoch in range(init_epoch, args.max_epoch + 1):
        # refine the step-size
        if epoch in args.schedule:
            initial_lr *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr
        
        model.train()
        tl = Averager()
        ta = Averager()
        print('len of train loader', len(train_loader))
        for i, batch in enumerate(train_loader_categories, 1):
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, label, ids = batch[0].cuda(), batch[1].cuda(), batch[2]
                label = label.type(torch.cuda.LongTensor)
            else:
                data, label, ids = batch
                label = label.type(torch.LongTensor)
            # print('HEREEEEEE')
            # print('data', [data.shape, data.min(), data.max()])
            with torch.no_grad():
                # print('before ids ', ids)
                # print(label)
                features = model.encoder(data)
            # print('features ', features.shape)
            embeds = model._forward(features, ids=ids)[0]
            # print('embeds ', embeds.shape)
            # print('here train',[embeds.shape, features.shape])
            
            loss = criterion(embeds, features)
            # print(loss)
            # acc = count_acc(logits, label)
            writer.add_scalar('data/loss', float(loss), global_count)
            # writer.add_scalar('data/acc', float(acc), global_count)
            if (i-1) % 100 == 0:
                print('epoch {}, train {}/{}, loss={:.4f}'.format(epoch, i, len(train_loader), loss.item()))

            tl.add(loss.item())
            # ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # break

        tl = tl.item()
        # ta = ta.item()

        # do not do validation in first 500 epoches
        if epoch > 100 or (epoch-1) % 5 == 0:
            model.eval()
            vl = Averager()
         
            print('[Dist] best epoch {}, current best val loss={:.4f}'.format(trlog['min_loss_epoch'], trlog['min_loss']))

            label = torch.arange(valset.num_class).repeat(args.query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)        
            with torch.no_grad():
                for i, batch in tqdm(enumerate(val_loader, 1)):
                    if torch.cuda.is_available():
                        data, label = batch[0].cuda(), batch[1].cuda()
                        label = label.type(torch.cuda.LongTensor)
                    else:
                        data, label = batch
                        label = label.type(torch.LongTensor)
                    # print('val data ', data.shape)
                    features = model.encoder(data)
                    # print('features before ', features.shape)
                    embeds = model._forward(features)

                    # print('embeds ', embeds.shape)
                    
                    # print('here',[embeds.shape, features.shape])
                    val_loss = criterion(embeds, features)
                    vl.add(val_loss.item())
                    # break
            vl = vl.item()
         
            writer.add_scalar('data/val_loss', float(vl), epoch)
           
            print('epoch {}, val, loss={:.4f}'.format(epoch, vl))
    
            if vl < trlog['min_loss']:
                trlog['min_loss'] = vl
                trlog['min_loss_epoch'] = epoch
                save_model('min_loss')
                save_checkpoint(True)
                      
    
            trlog['train_loss'].append(tl)
            trlog['val_loss'].append(vl)
            save_model('epoch-last')
    
            print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
    writer.close()
    
    
    import pdb
    pdb.set_trace()