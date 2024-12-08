import torch.nn as nn

from isegm.utils.serialization import serialize
from .is_model import ISModel
from isegm.model.modifiers import LRMult
from .modeling.u2net import U2NETP

class U2netModel(ISModel):
    @serialize
    def __init__(self,ocr = False, ocr_width=256, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.feature_extractor = U2NETP(3,1) # feature extractor -- u2net lightweight version
        self.feature_extractor.apply(LRMult(backbone_lr_mult))


    def backbone_forward(self, image, coord_features=None):
        net_outputs = self.feature_extractor(image, coord_features)

        # return {'instances': net_outputs[0], 'instances_aux': net_outputs[1]}
        return {'instances': net_outputs[0], 'instances_aux': net_outputs[1]}
