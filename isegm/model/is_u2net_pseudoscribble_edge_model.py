import torch.nn as nn
from isegm.utils.serialization import serialize
from .is_model import ISModel
from isegm.model.modifiers import LRMult
from .modeling.multitask_u2net_p_edge_generation import PseudoEdgeScribbleU2NETP


class U2netEdgePseudoScribbleModel(ISModel):
    @serialize
    def __init__(self, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.feature_extractor = PseudoEdgeScribbleU2NETP(3,1) # Feature Extractor -- U2Net Lightweight Version
        self.feature_extractor.apply(LRMult(backbone_lr_mult))


    def backbone_forward(self, image, coord_features=None):
        net_outputs = self.feature_extractor(image, coord_features)

        return {'instances': net_outputs[0], 'instances_aux': net_outputs[1], 'fpmap': net_outputs[2],
                'fpmap_aux': net_outputs[3], 'fnmap': net_outputs[4], 'fnmap_aux': net_outputs[5]}
