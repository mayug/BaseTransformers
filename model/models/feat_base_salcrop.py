from .feat_baseclass2 import FEATBaseClass2, get_k_base, ids2classes
import torch
import numpy as np


class FEATBaseSalCrop(FEATBaseClass2):
    def __init__(self, args):
        super().__init__(args)
        # type of saliency crop
        self.sal_crop = args.sal_crop
        self.cropped_base_embeds = False
        if self.cropped_base_embeds:
            proto_dict = torch.load('/home/mayug/projects/few_shot/notebooks/embeds_cache_crop2.pt')
            self.all_proto = proto_dict['embeds'].cuda()
            self.proto_dict = proto_dict
            self.all_classes = ids2classes(np.array(proto_dict['ids']))

        # another experiment changing
        # proto_dict and al_proto to embeds_cache_crop2.pt
