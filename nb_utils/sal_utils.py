from nb_utils.vis_utils import get_class

import sys
sys.path.append('/home/mayug/projects/FastSal')
  
import model1.fastSal as fastsal
from utils import load_weight
import torch
import numpy as np
import cv2
import cv2 as cv
import torchvision.transforms as tfms
import matplotlib.pyplot as plt
import torch.nn.functional as F

coco_c = 'weights/coco_C.pth'  # coco_C
coco_a = '/home/mayug/projects/FastSal/weights/coco_A.pth'  # coco_A
salicon_c = 'weights/salicon_C.pth'  # salicon_C
salicon_a = 'weights/salicon_A.pth'  # coco_A

model = fastsal.fastsal(pretrain_mode=False, model_type='A')
model = model.eval()
model = model.cuda()
state_dict, opt_state = load_weight(coco_a, remove_decoder=False)
model.load_state_dict(state_dict)

print('__name__ ', __name__)

    
def resize(tens, shape=(256,192)):
    if isinstance(tens, torch.Tensor):
        tens = tens.cpu().numpy()
    if tens.ndim==4:
        a = torch.Tensor([np.transpose(cv2.resize(np.transpose(t, axes=(1,2,0)), dsize=shape), (2,0,1)) for t in tens])
    elif tens.ndim==3:
        a = torch.Tensor([cv2.resize(t, dsize=shape) for t in tens])
    return a


def minmax(img):
    img = (img- img.min())/(img.max()-img.min())
    return img


def get_saliency_crops(data, return_masks=False, use_minmax=True):
    res = tfms.Resize(size=(192,256))
    y= res(data)
#     print(['y', y.min(), y.max(), y.shape])
    with torch.no_grad():
        z = model(y)
#     print(['z', z.min(), z.max(), z.shape])
    
    z = minmax(torch.repeat_interleave(z.detach(),3,dim=1))
#     print(['z', z.min(), z.max(), z.shape])
    z_binary = z>0.6
#     print(z.shape)
    cropped = y*z_binary
    if use_minmax:
        minmax_out = minmax
    else:
        minmax_out = lambda x: x
    
    if return_masks:
        return resize(minmax_out(cropped), (84,84)), resize(z_binary, (84,84))
    return resize(minmax_out(cropped), (84,84))



def get_sal(data, out_size=(84, 84)):
    # print(['data', data.min(), data.max(), data.shape])
    res = tfms.Resize(size=(192,256))
    y= res(data)
    # print(['y', y.min(), y.max(), y.shape])
    with torch.no_grad():
        z = model(y)
    # print(['z', z.min(), z.max(), z.shape])

    z = minmax(z)

    # z= F.sigmoid(z)

    res = tfms.Resize(size=out_size)
    y= res(z)
    return y


def plot_sal_crops(data, data_crops, target):
    for img, crop, t in zip(data, data_crops, target):
    
        img = np.transpose(img.detach().cpu().numpy(), axes=(1,2,0))
        crop = np.transpose(crop.detach().cpu().numpy(), axes=(1,2,0))

        print([t, get_class(t,'test')])
        plt.figure()

#         print(['hee ', img.shape, crop.shape])
        plt.imshow(minmax((img)))
        plt.show()

        plt.figure()
        plt.imshow(crop)
        plt.show()

def get_samples(trainer, loader, n=200, saliency=False, return_inputs=False):
    model = trainer.model.eval()
    inputs = [] # these are query examples in eval stage
    preds = []  # these are query examples in eval stage
    samples_before = [] # these are all examples in eval stage
    samples_feat = [] # these are support examples in eval stage
    targets = []# these are all examples in 
    targets_feat = []# these are support examples in eval stage
    targets_query = []

    samples_before_cropped = []
    samples_feat_cropped = []
    
    
    for i in range(n):
        data, target = next(iter(loader))
        with torch.no_grad():
            print('i', i)
            s_idx, q_idx = model.split_instances(None)
#             print('here ', q_idx.squeeze().view(-1))
#             asd
            inputs.append(data[q_idx.squeeze().view(-1)])
            t_s = target[s_idx.squeeze().view(-1)]
            t_q = target[q_idx.squeeze().view(-1)]
            targets_query.append(t_q)
    
            data = data.cuda()
            
            s_b, s_f, t, t_f, l = _get_samples(data, model, True)
            
#             print('s_b', s_b.shape)
#             print('logits', l.shape)
#             asd
            
            samples_before.append(s_b)
            samples_feat.append(s_f) # support examples
            targets.append(t)
            targets_feat.append(t_f) # same as query if numbers are equal
#             print('t_s ', t_s)
#             print('l.argmax ', l.argmax(axis=1).shape)
#             print('here ', t_s[l.argmax(axis=1)].shape)
#             asd
            preds.append(t_s[l.argmax(axis=1)])
            
            out = [samples_before, samples_feat, targets, targets_feat]
            # saliency
            if saliency:
                data_cropped = get_saliency_crops(data)

                data_cropped = resize(data_cropped, shape=(84,84)).cuda()
#
                s_b_c, s_f_c, _, _ = _get_samples(data_cropped, model, True)
                
                samples_before_cropped.append(s_b_c)
                samples_feat_cropped.append(s_f_c)
                
                out.extend([samples_before_cropped, samples_feat_cropped])
                
    for o in out:
        print([len(o), type(o)])
    
    if return_inputs:
        out.extend([inputs, preds, targets_query])

    
    return [np.concatenate(o) for o in out]



def _get_samples(data, model, targets, return_logits=False):
    
    with torch.no_grad():
        features = model.encoder(data)
    
    features = features.unsqueeze(0)
    samples_before = features.detach().cpu().squeeze().numpy()
#     targets = targets.numpy()

    # feat;
    with torch.no_grad():
        logits = model(data)[0]


#     print('after_attn shape', model.after_attn.shape)
    support_idx, query_idx = model.split_instances(data)
    samples_feat = model.after_attn.squeeze().detach().cpu().squeeze().numpy()

    targets_feat = targets[support_idx.squeeze()].numpy()

    if return_logits:
        return samples_before, samples_feat, targets, targets_feat, logits.detach().cpu().numpy()

    return samples_before, samples_feat, targets, targets_feat

def get_saliency_crops2(data, return_masks=False, fill_mask=False, use_minmax=True, thresh=0.6):
    res = tfms.Resize(size=(192,256))
    out_res = tfms.Resize(size=(84,84))
    y= res(data)
#     print(['y', y.min(), y.max(), y.shape])
    with torch.no_grad():
        z = model(y)
#     print(['z', z.min(), z.max(), z.shape])
    
    z = minmax(torch.repeat_interleave(z.detach(),3,dim=1))
#     z = minmax(z.squeeze())
#     print(['z', z.min(), z.max(), z.shape])
    z_binary = z>thresh
#     print(z.shape)
#     cropped = y*z_binary
    
#     cropped = y
#     cropped[z_binary] = 0
    
    temp = y
    
    color_means = temp.mean([2,3])
#     print(color_means.shape)
    color_means_2d = torch.zeros_like(temp)
    color_means_2d = color_means.unsqueeze(2).repeat_interleave(192, dim=2).unsqueeze(3).repeat_interleave(256, dim=3)
    
#     print(color_means_2d[0,0,:,:].mean(), color_means[0,0])
    
#     print(['here ', temp.shape, z_binary.shape, z_binary[:,0,:,:].shape])
    
    
    temp[~z_binary] = color_means_2d[~z_binary]

    
    
    
    
    cropped = temp
#     return cropped

    if use_minmax:
        minmax_out = minmax
    else:
        minmax_out = lambda x: x

    if return_masks:
        return out_res(minmax_out(cropped)), out_res(z_binary)
    return out_res(minmax_out(cropped))


def rectangle_crops(img, sal):
    resize_imgs = []
    for s,im in zip(sal, img):
        x,y,w,h = get_biggest_rectangle(s.squeeze().numpy())
#         print('im', im.shape)
        crop_img = im[:, x:x+w, y:y+h]
        res = tfms.Resize((84,84))
#         print(crop_img.shape, type(crop_img))
        crop_resize_img = res(crop_img)
        resize_imgs.append(crop_resize_img.unsqueeze(0))

    return torch.cat(resize_imgs, dim=0).numpy()

def get_biggest_rectangle(sal):
    # input numpy float array 0-1
    img = (sal*255).astype(np.uint8)
    th = int(0.6*(255))
    ret, thresh = cv.threshold(img, th, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    return x,y,w,h    


def get_saliency_crops3(data, return_masks=False, fill_mask=False, use_minmax=True):
    
    # rectangle crops and resize
    
    sal = get_sal(data)
    cropped = rectangle_crops(data,sal)
    if use_minmax:
        minmax_out = minmax
    else:
        minmax_out = lambda x: x
    
    return minmax_out(cropped)


    