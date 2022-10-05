import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer, get_update_loader
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
    AccuracyClassAverager
)
from tensorboardX import SummaryWriter
import wandb
from collections import deque
from tqdm import tqdm
import json
from shutil import copyfile
from json2html import *
from apex import amp
def get_label_aux(n_batch):
    label_aux = torch.cat([torch.arange(start=n_batch-2, end=2*n_batch-2),
        torch.arange(start=0, end=n_batch)])

    return label_aux

from apex import amp

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)


        self.return_simclr = True if args.return_simclr is not None else False
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)
        self.pass_ids = bool(args.pass_ids) # to remove same class instances during training using base dataset
        self.mixed_precision = args.mixed_precision
        if self.mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                opt_level=self.mixed_precision)
        
        if args.update_base_embeds:
            batch_size = 32
            self.update_loader = get_update_loader(args, batch_size)



    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        
        if args.model_class == 'FEATBaseTransformer3_Aux':
            label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.k-1)
        else:
            if args.label_aux_type is None:
                label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
            elif args.label_aux_type == 'random':
                label_aux = torch.randperm(args.way, dtype=torch.int8).repeat(args.shot + args.query)

            if args.simclr_loss_type=='ver2.1':
                label_aux = get_label_aux((args.shot+args.query)*(args.way))

        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        # print('label_aux ', label_aux)
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux
    
    def _fix_BN_only(self, model):
        for module in model.encoder.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()

    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        elif self.args.fix_BN_only:
            self._fix_BN_only(self.model)
        
        # start FSL training
        label, label_aux = self.prepare_label()

        # print('before ver2.2 ', label_aux)

        print('Using mixed precision with opt level = ', self.mixed_precision)
        # asd

            

        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            elif self.args.fix_BN_only:
                self._fix_BN_only(self.model)
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()
            if self.args.dataset == 'MiniImageNet':
                train_classes = 64
            elif self.args.dataset == 'CUB':
                train_classes = 100
            elif self.args.dataset == 'TieredImageNet' or self.args.dataset == 'TieredImageNet_og':
                train_classes = 351
            tca = AccuracyClassAverager(train_classes)

            start_tm = time.time()

            if self.args.update_base_interval is not None:
                if epoch%self.args.update_base_interval==0:
                    print('running base proto update')
                    self.model.update_base_protos()
            
            if self.args.update_base_embeds:
                if self.trlog['max_acc_epoch']==epoch-1 and epoch>=args.patience:
                    self.model.update_2d_embeds(self.update_loader)

            for batch in self.train_loader:
                self.optimizer.zero_grad()


                self.train_step += 1

                if torch.cuda.is_available():
 
                    if self.pass_ids:
                        if self.return_simclr:
                            data, gt_label, ids, data_simclr = batch[0].cuda(), batch[1].cuda(), batch[2], batch[3].cuda()
                        else:
                            data, gt_label, ids = batch[0].cuda(), batch[1].cuda(), batch[2]
                    else:
                        if self.return_simclr:
                            data, gt_label, data_simclr = [_.cuda() for _ in batch]
                        else:
                            data, gt_label = [_.cuda() for _ in batch]
                else:
                    if self.pass_ids:
                        if self.return_simclr:
                            data, gt_label, ids, data_simclr = batch[0], batch[1], batch[2], batch[3]
                        else:
                            data, gt_label, ids = batch[0], batch[1], batch[2]
                    else:
                        if self.return_simclr:
                            data, gt_label, data_simclr = batch[0], batch[1], batch[2]
                        else:
                            data, gt_label = batch[0], batch[1]

                # print('data_simclr ', [data_simclr.shape, data_simclr.min(), data_simclr.max()])
                
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)
                
                if self.pass_ids:
                    # logits, reg_logits = self.para_model(data, ids) # unhide this for prev functionality
                    if self.return_simclr:
                        logits, reg_logits = self.model(data, ids, simclr_images=data_simclr)
                    else:
                        logits, reg_logits = self.para_model(data, ids)
                        # logits, reg_logits = self.model(data, ids)
                else:
                    # logits, reg_logits = self.para_model(data)
                    if self.return_simclr:
                        logits, reg_logits = self.model(data, simclr_images=data_simclr)
                    else:
                        logits, reg_logits = self.model(data)

                aux_loss = None
                if reg_logits is not None:
                    loss = F.cross_entropy(logits, label)
                    if self.return_simclr:
                        if self.args.simclr_loss_type == 'ver1':
                        
                                # print('reg_logits ', [reg_logits.shape,reg_logits.min(), reg_logits.max()])
                                # print(reg_logits)
                                reg_logits = F.log_softmax(reg_logits, dim=2)
                                # print('after softmax ', [reg_logits.shape,reg_logits.min(), reg_logits.max()])
                                # print(label_aux)
                                reg_logits = reg_logits.reshape(-1, self.args.way)
                                # print('reg logits after reshape ', reg_logits.shape)
                                aux_loss = F.nll_loss(reg_logits, label_aux, reduction='mean')
                                

                                total_loss = loss + args.balance * aux_loss
                                # print('loss, aux_loss')
                                # print([loss, aux_loss])
                                # asd
                                # print('total ', total_loss)
                                # asd
                        elif self.args.simclr_loss_type == 'ver2.1':
                            aux_loss = F.cross_entropy(reg_logits, label_aux)
                            # print('loss, aux_loss ', [loss, aux_loss])
                            total_loss = loss + args.balance * aux_loss
                            # asd]
                        elif self.args.simclr_loss_type == 'ver2.2':
                            # print('self.model.label_aux ', self.model.label_aux)
                            # asd

                            aux_loss = F.cross_entropy(reg_logits, self.model.label_aux)
                            # print('loss, aux_loss ', [loss, aux_loss])
                            total_loss = loss + args.balance * aux_loss

                    
                    
                    else:
                        total_loss = (1-args.balance)*loss + args.balance * self.loss(reg_logits, label_aux)
                else:
                    # loss = self.loss(logits, label)
                    # total_loss = self.loss(logits, label)
                    loss = F.cross_entropy(logits, label)
                    total_loss = F.cross_entropy(logits, label)
                tl2.add(loss)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)

                tl1.add(total_loss.item())
                ta.add(acc)
                self.try_logging(total_loss, loss, acc, aux_loss=aux_loss)
                # tca.add(logits.cpu(), gt_label[self.args.way:].cpu())

                # self.optimizer.zero_grad()
                # total_loss.backward() # unhide this for non mixed precision
                if self.mixed_precision:
                    with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()


                # step optimizer
                # self.optimizer.step()

                # if 'cycl' in self.args.lr_scheduler:
                #     # print('steppugn inside batch loop')
                #     self.lr_scheduler.step() 

                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)
                
                self.optimizer.step() 


                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)

                # refresh start_tm
                start_tm = time.time()
                
            if 'cycl' not in self.args.lr_scheduler:
                print('stepping outside batch loop')
                self.lr_scheduler.step()
      
            print('logits range ', [logits.min().item(), logits.max().item()])
            # print('class wise test accuracies ')
            # print(tca.item())
            self.try_evaluate(epoch)



            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )


        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')
        

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))

        vca = AccuracyClassAverager(16)

        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    if self.pass_ids:
                        data, gt_label, ids = batch[0].cuda(), batch[1].cuda(), batch[2]
                    else:
                        data, gt_label = [_.cuda() for _ in batch]
                else:
                    if self.pass_ids:
                        data, gt_label, ids = batch[0], batch[1], batch[2]
                    else:
                        data, gt_label = batch[0], batch[1]
                if self.pass_ids:
                    logits = self.model(data, ids)
                else:
                    logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                # vca.add(logits.cpu(), gt_label[self.args.way:].cpu())
                
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        # print('class wise test accuracies ')
        # print(vca.item())
        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        elif self.args.fix_BN_only:
            self._fix_BN_only(self.model)

        return vl, va, vap

    def load_model(self):
        args = self.args
        path = args.test
        print('setting state dict of model to {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['params'])
        if self.mixed_precision is not None:
            print('setting amp state dict ')
            amp.load_state_dict(checkpoint['amp'])

    def save_model(self, name):
        if self.mixed_precision is not None:
            save_dict = dict(params=self.model.state_dict(),
            amp=amp.state_dict())
        else:
            save_dict = dict(params=self.model.state_dict())

        torch.save(
            save_dict,
            osp.join(self.args.save_path, name + '.pth')
        )

    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        if args.test:
            path = args.test
        else:
            path = osp.join(self.args.save_path, 'max_acc.pth')
        self.model.load_state_dict(torch.load(path)['params'])
        self.model.eval()
        record = np.zeros((10000, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        if self.args.dataset == 'MiniImageNet':
            test_classes = 20
        elif self.args.dataset == 'CUB':
            test_classes = 50
        elif self.args.dataset == 'TieredImageNet' or self.args.dataset == 'TieredImageNet_og':
            test_classes = 160
        tca = AccuracyClassAverager(test_classes)

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                # if torch.cuda.is_available():
                #     data, gt_label = [_.cuda() for _ in batch]
                # else:
                #     data = batch[0]
                if torch.cuda.is_available():
                    if self.pass_ids:
                        data, gt_label, ids = batch[0].cuda(), batch[1].cuda(), batch[2]
                    else:
                        data, gt_label = [_.cuda() for _ in batch]
                else:
                    if self.pass_ids:
                        data, gt_label, ids = batch[0], batch[1], batch[2]
                    else:
                        data, gt_label = batch[0], batch[1]
                if self.pass_ids:
                    logits = self.model(data, ids)
                else:
                    logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                # print([data.shape, gt_label.shape])
                # print('here')
                # print(logits.cpu().shape)
                # print(gt_label[self.args.eval_shot*self.args.way:].cpu().shape)
                # asd
                tca.add(logits.cpu(), gt_label[self.args.eval_shot*self.args.way:].cpu())
       

     
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))

        print('class wise test accuracies ')
        print(tca.item())
        if args.test:
            pkl_path = osp.dirname(args.test)
        else:
            pkl_path = args.save_path
        
        save_dict = tca.item()
        save_dict['max_acc_epoch'] = self.trlog['max_acc_epoch']
        save_dict['best_val_acc'] = self.trlog['max_acc']
        save_dict['test_acc'] = self.trlog['test_acc']
        json.dump(save_dict, open(osp.join(pkl_path, 'test_class_acc.json'), 'w'))
        print('saved class accuracy json in {}'.format(pkl_path))

        print('Saving class accuracy json in wandb as a file')
        # artifact = self.wandb_run.Artifact('test_accuracies', type='test_acc_json')
        # artifact.add_file(pkl_path)
        # self.wandb_run.log_artifact(artifact)
        try_count = 0
        while try_count<10:
            try:
                self.wandb_run.save(osp.join(pkl_path, 'test_class_acc.json'))
                break
            except Exception as ex:
                try_count = try_count + 1
                print([ex, 'Retrying'])
                time.sleep(10)
            
        print('here    ')
        print(save_dict)
        self.wandb_run.log({"accuracies": wandb.Html(json2html.convert(json = save_dict))})
        # copyfile(osp.join(pkl_path, 'test_class_acc.json'),
        #     osp.join(self.wandb_run.dir, 'test_class_acc.json'))

        self.wandb_run.finish()
        return vl, va, vap
    
    def final_record(self):
        # save the best performance in a txt file
        
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))            