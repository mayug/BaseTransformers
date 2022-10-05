
import os.path as osp
import PIL
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os


THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH1, 'data/cub/images')
SPLIT_PATH = osp.join(ROOT_PATH2, 'data/cub/split')
CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')

from SimCLR.data_aug.view_generator import ContrastiveLearningViewGenerator

def identity(x):
    return x


def get_simclr_pipeline_transform(size, s=1, extra_transforms=None):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.1 * s, 0.1 * s, 0.1 * s, 0.1 * s)
    # print('here ', extra_transforms)
    data_transforms = torch.nn.Sequential(
        transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                        #   transforms.RandomRotation(30, interpolation=transforms.InterpolationMode.BILINEAR),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                        #   transforms.RandomGrayscale(p=0.2),
                                          *extra_transforms
#                                           GaussianBlur(kernel_size=int(0.1 * size)),
#                                           transforms.ToTensor()
    )
    print('simclr transforms are ')
    print(data_transforms)
    return data_transforms
# This is for the CUB dataset
# It is notable, we assume the cub images are cropped based on the given bounding boxes
# The concept labels are based on the attribute value, which are for further use (and not used in this work)

class CUB(Dataset):

    def __init__(self, setname, args, augment=False, return_id=False, 
                return_simclr=None):
        im_size = args.orig_imsize
        txt_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]
        cache_path = osp.join( CACHE_PATH, "{}.{}.{}.pt".format(self.__class__.__name__, setname, im_size) )

        self.use_im_cache = ( im_size != -1 ) # not using cache
        self.return_id = return_id
        if self.use_im_cache:
            if not osp.exists(cache_path):
                print('* Cache miss... Preprocessing {}...'.format(setname))
                resize_ = identity if im_size < 0 else transforms.Resize(im_size)
                data, label = self.parse_csv(txt_path)
                self.path = data
                self.data = [ resize_(Image.open(path).convert('RGB')) for path in data ]
                self.label = label
                print('* Dump cache from {}'.format(cache_path))
                torch.save({'data': self.data, 'label': self.label, 'path': self.path}, cache_path)
            else:
                print('* Load cache from {}'.format(cache_path))
                cache = torch.load(cache_path)
                self.data  = cache['data']
                self.label = cache['label']
                self.path = cache['path']
        else:
            self.data, self.label = self.parse_csv(txt_path)
        
        self.num_class = np.unique(np.array(self.label)).shape[0]
        if args.image_size is not None:
            image_size = args.image_size
        else:
            image_size = 84
        
        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            print('image_size is ', image_size)
            print('first resize is ', int(image_size*(84/84)))
            transforms_list = [
                  transforms.Resize(int(image_size*(84/84))),
                  transforms.CenterCrop(image_size), 
                  transforms.ToTensor(),
                ]

        # Transformation
        if args.backbone_class == 'ConvNet':
            self.norm = transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))

        elif args.backbone_class == 'Res12':
            self.norm = transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))

        elif args.backbone_class == 'Res18':
            self.norm = transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))

        elif args.backbone_class == 'WRN':
            self.norm = transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))

        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')
        
        self.transform = transforms.Compose(
            transforms_list + [ self.norm
            
        ])
        self.return_simclr = return_simclr

        if self.return_simclr is not None:
            print('CUB dataset with return_simclr = ', return_simclr)
            extra_transforms = [ transforms.Resize(int(image_size*(84/84))),
                  transforms.CenterCrop(image_size), 
                  self.norm]
            if self.use_im_cache: 
                self.init_transform = transforms.Compose([transforms.ToTensor()])
                self.simclr_transform = ContrastiveLearningViewGenerator(
                    get_simclr_pipeline_transform(im_size,
                    extra_transforms=extra_transforms), n_views=self.return_simclr)
            else:
                self.init_transform = transforms.Compose([transforms.Resize(128),
                    transforms.ToTensor()])
                self.simclr_transform = ContrastiveLearningViewGenerator(
                    get_simclr_pipeline_transform(128,
                    extra_transforms=extra_transforms), n_views=self.return_simclr)

    def parse_csv(self, txt_path):
        data = []
        label = []
        lb = -1
        self.wnids = []
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        for l in lines:
            context = l.split(',')
            name = context[0] 
            wnid = context[1]
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                
            data.append(path)
            label.append(lb)

        return data, label


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]

        if self.use_im_cache:
            image = data
        else:
            image = Image.open(data).convert('RGB')


        if self.return_simclr is not None:
            # print('image ', self.init_transform(image).shape, self.init_transform(image).min(), self.init_transform(image).max())
            simclr_images = self.simclr_transform(self.init_transform(image))
            simclr_images = [s.unsqueeze(0) for s in simclr_images]
            simclr_images = torch.cat(simclr_images, dim=0)
        
        # print('transform before applying ', self.transform)
        image = self.transform(image)

        if self.return_id:
            if self.use_im_cache:
                path = self.path[i]
                if self.return_simclr is not None:
                    return image, label, os.path.basename(path)[:-len('.jpg')], simclr_images
                else:
                    return image, label, os.path.basename(path)[:-len('.jpg')]
            else:
                if self.return_simclr is not None:
                    return image, label, os.path.basename(data)[:-len('.jpg')], simclr_images
                else:
                    return image, label, os.path.basename(data)[:-len('.jpg')]
        if self.return_simclr is not None:
            # print('here ', simclr_images.shape)
            return image, label, simclr_images
        return image, label


    def get_image(self, id):
        i = self.csv[self.csv['filename']==f'{id}.jpg'].index.item()
        return self.__getitem__(i)            


# below is default

# import os.path as osp
# import PIL
# from PIL import Image

# import numpy as np
# from torch.utils.data import Dataset
# from torchvision import transforms
# import torch
# import os

# THIS_PATH = osp.dirname(__file__)
# ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
# ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
# IMAGE_PATH = osp.join(ROOT_PATH1, 'data/cub/images')
# SPLIT_PATH = osp.join(ROOT_PATH2, 'data/cub/split')
# CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')

# # This is for the CUB dataset
# # It is notable, we assume the cub images are cropped based on the given bounding boxes
# # The concept labels are based on the attribute value, which are for further use (and not used in this work)

# class CUB(Dataset):

#     def __init__(self, setname, args, augment=False, return_id=False, return_simclr=None):
#         im_size = args.orig_imsize
#         txt_path = osp.join(SPLIT_PATH, setname + '.csv')
#         lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]
#         cache_path = osp.join( CACHE_PATH, "{}.{}.{}.pt".format(self.__class__.__name__, setname, im_size) )

#         self.use_im_cache = ( im_size != -1 ) # not using cache
#         self.return_id = return_id
#         if self.use_im_cache:
#             if not osp.exists(cache_path):
#                 print('* Cache miss... Preprocessing {}...'.format(setname))
#                 resize_ = identity if im_size < 0 else transforms.Resize(im_size)
#                 data, label = self.parse_csv(txt_path)
#                 self.path = data
#                 self.data = [ resize_(Image.open(path).convert('RGB')) for path in data ]
#                 self.label = label
#                 print('* Dump cache from {}'.format(cache_path))
#                 torch.save({'data': self.data, 'label': self.label, 'path': self.path}, cache_path)
#             else:
#                 print('* Load cache from {}'.format(cache_path))
#                 cache = torch.load(cache_path)
#                 self.data  = cache['data']
#                 self.label = cache['label']
#                 self.path = cache['path']
#         else:
#             self.data, self.label = self.parse_csv(txt_path)
        
#         self.num_class = np.unique(np.array(self.label)).shape[0]
#         image_size = 84
        
#         if augment and setname == 'train':
#             transforms_list = [
#                   transforms.RandomResizedCrop(image_size),
#                   transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#                   transforms.RandomHorizontalFlip(),
#                   transforms.ToTensor(),
#                 ]
#         else:
#             transforms_list = [
#                   transforms.Resize(92),
#                   transforms.CenterCrop(image_size),
#                   transforms.ToTensor(),
#                 ]

#         # Transformation
#         if args.backbone_class == 'ConvNet':
#             self.transform = transforms.Compose(
#                 transforms_list + [
#                 transforms.Normalize(np.array([0.485, 0.456, 0.406]),
#                                      np.array([0.229, 0.224, 0.225]))
#             ])
#         elif args.backbone_class == 'Res12':
#             self.transform = transforms.Compose(
#                 transforms_list + [
#                 transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
#                                      np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
#             ])
#         elif args.backbone_class == 'Res18':
#             self.transform = transforms.Compose(
#                 transforms_list + [
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])            
#         elif args.backbone_class == 'WRN':
#             self.transform = transforms.Compose(
#                 transforms_list + [
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])         
#         else:
#             raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

#     def parse_csv(self, txt_path):
#         data = []
#         label = []
#         lb = -1
#         self.wnids = []
#         lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

#         for l in lines:
#             context = l.split(',')
#             name = context[0] 
#             wnid = context[1]
#             path = osp.join(IMAGE_PATH, name)
#             if wnid not in self.wnids:
#                 self.wnids.append(wnid)
#                 lb += 1
                
#             data.append(path)
#             label.append(lb)

#         return data, label


#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         data, label = self.data[i], self.label[i]
#         if self.use_im_cache:
#             image = self.transform(data)
#         else:
#             image = self.transform(Image.open(data).convert('RGB'))
#         # print('self.return_id ', self.return_id)
#         if self.return_id:
#             if self.use_im_cache:
#                 path =self.path[i]
#                 return image, label, os.path.basename(path)[:-len('.jpg')]
#             else:
#                 return image, label, os.path.basename(data)[:-len('.jpg')]
#         return image, label  












# below is old simclr


# import os.path as osp
# import PIL
# from PIL import Image
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# import os

# THIS_PATH = osp.dirname(__file__)
# ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
# ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
# IMAGE_PATH = osp.join(ROOT_PATH1, 'data/cub/images')
# SPLIT_PATH = osp.join(ROOT_PATH2, 'data/cub/split')
# CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')

# from SimCLR.data_aug.view_generator import ContrastiveLearningViewGenerator

# def identity(x):
#     return x


# def get_simclr_pipeline_transform(size, s=1, extra_transforms=None):
#     """Return a set of data augmentation transformations as described in the SimCLR paper."""
#     color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
#     print('here ', extra_transforms)
#     data_transforms = torch.nn.Sequential(
#         transforms.RandomResizedCrop(size=size),
#                                           transforms.RandomHorizontalFlip(),
#                                           transforms.RandomApply([color_jitter], p=0.8),
#                                           transforms.RandomGrayscale(p=0.2),
#                                           *extra_transforms
# #                                           GaussianBlur(kernel_size=int(0.1 * size)),
# #                                           transforms.ToTensor()
#     )
#     return data_transforms
# # This is for the CUB dataset
# # It is notable, we assume the cub images are cropped based on the given bounding boxes
# # The concept labels are based on the attribute value, which are for further use (and not used in this work)

# class CUB(Dataset):

#     def __init__(self, setname, args, augment=False, return_id=False, 
#                 return_simclr=None):
#         im_size = args.orig_imsize
#         txt_path = osp.join(SPLIT_PATH, setname + '.csv')
#         lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]
#         cache_path = osp.join( CACHE_PATH, "{}.{}.{}.pt".format(self.__class__.__name__, setname, im_size) )

#         self.use_im_cache = ( im_size != -1 ) # not using cache
#         self.return_id = return_id
#         if self.use_im_cache:
#             if not osp.exists(cache_path):
#                 print('* Cache miss... Preprocessing {}...'.format(setname))
#                 resize_ = identity if im_size < 0 else transforms.Resize(im_size)
#                 data, label = self.parse_csv(txt_path)
#                 self.path = data
#                 self.data = [ resize_(Image.open(path).convert('RGB')) for path in data ]
#                 self.label = label
#                 print('* Dump cache from {}'.format(cache_path))
#                 torch.save({'data': self.data, 'label': self.label, 'path': self.path}, cache_path)
#             else:
#                 print('* Load cache from {}'.format(cache_path))
#                 cache = torch.load(cache_path)
#                 self.data  = cache['data']
#                 self.label = cache['label']
#                 self.path = cache['path']
#         else:
#             self.data, self.label = self.parse_csv(txt_path)
        
#         self.num_class = np.unique(np.array(self.label)).shape[0]
#         if args.image_size is not None:
#             image_size = args.image_size
#         else:
#             image_size = 84
        
#         if augment and setname == 'train':
#             transforms_list = [
#                   transforms.RandomResizedCrop(image_size),
#                   transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#                   transforms.RandomHorizontalFlip(),
#                   transforms.ToTensor(),
#                 ]
#         else:
#             print('image_size is ', image_size)
#             print('first resize is ', int(image_size*(84/84)))
#             transforms_list = [
#                   transforms.Resize(int(image_size*(84/84))),
#                   transforms.CenterCrop(image_size), 
#                   transforms.ToTensor(),
#                 ]

#         # Transformation
#         if args.backbone_class == 'ConvNet':
#             self.norm = transforms.Normalize(np.array([0.485, 0.456, 0.406]),
#                                      np.array([0.229, 0.224, 0.225]))

#         elif args.backbone_class == 'Res12':
#             self.norm = transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
#                                      np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))

#         elif args.backbone_class == 'Res18':
#             self.norm = transforms.Normalize(np.array([0.485, 0.456, 0.406]),
#                                      np.array([0.229, 0.224, 0.225]))

#         elif args.backbone_class == 'WRN':
#             self.norm = transforms.Normalize(np.array([0.485, 0.456, 0.406]),
#                                      np.array([0.229, 0.224, 0.225]))

#         else:
#             raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')
        
#         self.transform = transforms.Compose(
#             transforms_list + [ self.norm
            
#         ])
#         self.return_simclr = return_simclr

#         if self.return_simclr is not None:
#             print('CUB dataset with return_simclr = ', return_simclr)
#             extra_transforms = [ transforms.Resize(int(image_size*(84/84))),
#                   transforms.CenterCrop(image_size), 
#                   self.norm]
#             if self.use_im_cache: 
#                 self.init_transform = transforms.Compose([transforms.ToTensor()])
#                 self.simclr_transform = ContrastiveLearningViewGenerator(
#                     get_simclr_pipeline_transform(im_size,
#                     extra_transforms=extra_transforms), n_views=self.return_simclr)
#             else:
#                 self.init_transform = transforms.Compose([transforms.Resize(128),
#                     transforms.ToTensor()])
#                 self.simclr_transform = ContrastiveLearningViewGenerator(
#                     get_simclr_pipeline_transform(128,
#                     extra_transforms=extra_transforms), n_views=self.return_simclr)

#     def parse_csv(self, txt_path):
#         data = []
#         label = []
#         lb = -1
#         self.wnids = []
#         lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

#         for l in lines:
#             context = l.split(',')
#             name = context[0] 
#             wnid = context[1]
#             path = osp.join(IMAGE_PATH, name)
#             if wnid not in self.wnids:
#                 self.wnids.append(wnid)
#                 lb += 1
                
#             data.append(path)
#             label.append(lb)

#         return data, label


#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         data, label = self.data[i], self.label[i]

#         if self.use_im_cache:
#             image = data
#         else:
#             image = Image.open(data).convert('RGB')


#         if self.return_simclr is not None:
#             # print('image ', self.init_transform(image).shape, self.init_transform(image).min(), self.init_transform(image).max())
#             simclr_images = self.simclr_transform(self.init_transform(image))
#             simclr_images = [s.unsqueeze(0) for s in simclr_images]
#             simclr_images = torch.cat(simclr_images, dim=0)
        
#         # print('transform before applying ', self.transform)
#         image = self.transform(image)

#         if self.return_id:
#             if self.use_im_cache:
#                 path = self.path[i]
#                 if self.return_simclr is not None:
#                     return image, label, os.path.basename(path)[:-len('.jpg')], simclr_images
#                 else:
#                     return image, label, os.path.basename(path)[:-len('.jpg')]
#             else:
#                 if self.return_simclr is not None:
#                     return image, label, os.path.basename(data)[:-len('.jpg')], simclr_images
#                 else:
#                     return image, label, os.path.basename(data)[:-len('.jpg')]
#         if self.return_simclr is not None:
#             print('here ', simclr_images.shape)
#             return image, label, simclr_images
#         return image, label


#     def get_image(self, id):
#         i = self.csv[self.csv['filename']==f'{id}.jpg'].index.item()
#         return self.__getitem__(i)            

