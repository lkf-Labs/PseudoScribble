import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x,coord_features):

        hx = x
        hxin = self.rebnconvin(hx)
        if coord_features is not None:
            hxin = hxin + coord_features

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin

def Norm_layer(norm_cfg, inplanes):
    norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
    if norm_cfg == 'BN':
        out = nn.BatchNorm2d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm2d(inplanes, **norm_op_kwargs)

    return out

def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out

class ConvNormNonlin(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1,
                 norm_cfg='BN', activation_cfg='ReLU'):
        super(ConvNormNonlin, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg)

    def forward(self, x):
        x = self.nonlin(self.norm(self.conv(x)))
        return x


# EGM
class Edge(nn.Module):
    """
        This repo holds code for Bea-net: body and edge aware network with multi-scale short-term concatenation for medical image segmentation (IEEE Journal of Biomedical and Health Informatics 2023)
    """
    def __init__(self, channels, layer=1, norm_cfg='BN', activation_cfg='ReLU'):
        super(Edge, self).__init__()
        if layer == 1:
            self.conv = ConvNormNonlin(channels * 3, channels, kernel_size=1, stride=1, padding=0,
                                       norm_cfg=norm_cfg, activation_cfg=activation_cfg)
        elif layer == 2:
            self.conv = ConvNormNonlin(channels * 2, channels, kernel_size=1, stride=1, padding=0,
                                       norm_cfg=norm_cfg, activation_cfg=activation_cfg)
        else:
            self.conv = ConvNormNonlin(channels, channels, kernel_size=1, stride=1, padding=0,
                                       norm_cfg=norm_cfg, activation_cfg=activation_cfg)
        self.layer = layer

    def forward(self, x):
        seg_edge0 = K.filters.sobel(x)


        x1 = F.max_pool2d(x, kernel_size=2)
        seg_edge1 = K.filters.sobel(x1)
        # seg_edge1 = F.interpolate(seg_edge1, scale_factor=(2, 2), mode='bilinear')
        seg_edge1 = _upsample_like(seg_edge1, seg_edge0)
        x2 = F.max_pool2d(x, kernel_size=4)
        seg_edge2 = K.filters.sobel(x2)
        # seg_edge2 = F.interpolate(seg_edge2, scale_factor=(4, 4), mode='bilinear')
        seg_edge2 = _upsample_like(seg_edge2, seg_edge0)
        seg_edge = seg_edge0
        if self.layer == 1:
            seg_edge = torch.cat([seg_edge0, seg_edge1, seg_edge2], dim=1)
        elif self.layer == 2:
            seg_edge = torch.cat([seg_edge0, seg_edge1], dim=1)

        seg_edge = self.conv(seg_edge)
        return seg_edge

# =============================================segmentation decoder=================================================================
class SegDecoder(nn.Module):

    def __init__(self, out_ch=1):
            super(SegDecoder, self).__init__()

            self.stage5d = RSU4F(128, 16, 64)
            self.stage4d = RSU4(128, 16, 64)
            self.stage3d = RSU5(128, 16, 64)
            self.stage2d = RSU6(128, 16, 64)
            self.stage1d = RSU7(128, 16, 64)

            self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
            self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
            self.side3 = nn.Conv2d(64,out_ch,3,padding=1)
            self.side4 = nn.Conv2d(64,out_ch,3,padding=1)
            self.side5 = nn.Conv2d(64,out_ch,3,padding=1)
            self.side6 = nn.Conv2d(64,out_ch,3,padding=1)

            self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, hx1, hx2, hx3, hx4, hx5, hx6):

        hx6up = _upsample_like(hx6, hx5)

        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))

        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))

        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))

        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))

        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1), None)

        #
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return [d0, (d0, d1, d2, d3, d4, d5, d6), (hx6, hx5d, hx4d, hx3d, hx2d, hx1d)]

# =============================================error prediction decoder（fpmap）======================================================
class FPMAPDecoder(nn.Module):
    """
    Introducing EGM to more effectively identify edge information
    """

    def __init__(self, out_ch=1):
            super(FPMAPDecoder, self).__init__()

            self.stage5d = RSU4F(128, 16, 64)
            self.stage4d = RSU4(128, 16, 64)
            self.stage3d = RSU5(128, 16, 64)
            self.stage2d = RSU6(128, 16, 64)
            self.stage1d = RSU7(128, 16, 64)

            self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

            self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

            self.edge1 = Edge(channels=64, layer=1, norm_cfg='BN', activation_cfg='ReLU')
            self.edge2 = Edge(channels=64, layer=1, norm_cfg='BN', activation_cfg='ReLU')
            self.edge3 = Edge(channels=64, layer=1, norm_cfg='BN', activation_cfg='ReLU')
            self.edge4 = Edge(channels=64, layer=1, norm_cfg='BN', activation_cfg='ReLU')
            self.edge5 = Edge(channels=64, layer=1, norm_cfg='BN', activation_cfg='ReLU')
            self.edge6 = Edge(channels=64, layer=1, norm_cfg='BN', activation_cfg='ReLU')

    def forward(self, aux_edge_input):
        hx6, hx5d, hx4d, hx3d, hx2d, hx1d = aux_edge_input

        hx6_edge = self.edge6(hx6)

        hx6_edge_up = _upsample_like(hx6_edge, hx5d)

        hx5d_edge = self.edge5(hx5d)

        d5 = self.stage5d(torch.cat([hx6_edge_up, hx5d_edge], dim=1))

        d5up = _upsample_like(d5, hx4d)

        hx4d_edge = self.edge4(hx4d)

        d4 = self.stage4d(torch.cat([d5up, hx4d_edge], dim=1))

        d4up = _upsample_like(d4, hx3d)

        hx3d_edge = self.edge3(hx3d)

        d3 = self.stage3d(torch.cat([d4up, hx3d_edge], dim=1))

        d3up = _upsample_like(d3, hx2d)

        hx2d_edge = self.edge2(hx2d)

        d2 = self.stage2d(torch.cat([d3up, hx2d_edge], dim=1))

        d2up = _upsample_like(d2, hx1d)

        hx1d_edge = self.edge1(hx1d)

        d1 = self.stage1d(torch.cat([d2up, hx1d_edge], dim=1), None)

        # Side output, used for calculating multiple losses
        d1 = self.side1(d1)

        d2 = self.side2(d2)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(d3)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(d4)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(d5)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6_edge)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))


        return [d0, (d0, d1, d2, d3, d4, d5)]


# =============================================error prediction decoder（fnmap）======================================================
class FNMAPDecoder(nn.Module):

    def __init__(self, out_ch=1):
            super(FNMAPDecoder, self).__init__()

            self.stage5d = RSU4F(128, 16, 64)
            self.stage4d = RSU4(128, 16, 64)
            self.stage3d = RSU5(128, 16, 64)
            self.stage2d = RSU6(128, 16, 64)
            self.stage1d = RSU7(128, 16, 64)

            self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

            self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, hx1, hx2, hx3, hx4, hx5, hx6):

        hx6up = _upsample_like(hx6, hx5)

        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))

        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))

        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))

        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))

        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1), None)

        # Side output, used for calculating multiple losses
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return [d0, (d0, d1, d2, d3, d4, d5, d6)]


### U2Net Lightweight Version for Multi-Task Learning -- Gland Segmentation with FpMap and FnMap ###

class PseudoEdgeScribbleU2NETP(nn.Module):
    """
    This repo holds code for U2-net: Going deeper with nested u-structure for salient object detection(Pattern recognition  2020).
    """
    def __init__(self,in_ch=3,out_ch=1):
        super(PseudoEdgeScribbleU2NETP,self).__init__()

        # shared encoder
        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(64,16,64)

        # Multi-Task Learning, three decoders
        # segmentation decoder
        self.segDecoder = SegDecoder(1)
        # FP decoder
        self.fpmapDecoder = FPMAPDecoder(1)
        # FN decoder
        self.fnmapDecoder = FNMAPDecoder(1)

    def forward(self,x,coord_features):
        # encoding
        fn_coord_features = None
        if coord_features.size()[1] > 64:
            # Not introducing prior masks for false positives and false negatives
            fn_coord_features = coord_features[:, :64, :, :]
            coord_features = coord_features[:, 64:, :, :]
        hx = x
        fn_hx = x.clone()

        #stage 1
        hx1 = self.stage1(hx,coord_features)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)

        # fn_hx1 = hx1
        # fn_hx2 = hx2
        # fn_hx3 = hx3
        # fn_hx4 = hx4
        # fn_hx5 = hx5
        # fn_hx6 = hx6

        # #===========================================FN encoding============================================================
        if fn_coord_features is not None:
            # stage 1
            fn_hx1 = self.stage1(fn_hx, fn_coord_features)
            fn_hx = self.pool12(fn_hx1)

            # stage 2
            fn_hx2 = self.stage2(fn_hx)
            fn_hx = self.pool23(fn_hx2)

            # stage 3
            fn_hx3 = self.stage3(fn_hx)
            fn_hx = self.pool34(fn_hx3)

            # stage 4
            fn_hx4 = self.stage4(fn_hx)
            fn_hx = self.pool45(fn_hx4)

            # stage 5
            fn_hx5 = self.stage5(fn_hx)
            fn_hx = self.pool56(fn_hx5)

            # stage 6
            fn_hx6 = self.stage6(fn_hx)


        # decoding
        seg_output, seg_side_output, aux_edge_input = self.segDecoder(hx1, hx2, hx3, hx4, hx5, hx6)
        if fn_coord_features is None:
            return [seg_output, seg_side_output, None, None, None, None]
        fp_output, fp_side_output = self.fpmapDecoder(aux_edge_input)
        # fn_output, fn_side_output = self.fnmapDecoder(hx1, hx2, hx3, hx4, hx5, hx6)
        fn_output, fn_side_output = self.fnmapDecoder(fn_hx1, fn_hx2, fn_hx3, fn_hx4, fn_hx5, fn_hx6)
        return [seg_output, seg_side_output, fp_output, fp_side_output, fn_output, fn_side_output]

    # Load pre-trained weights
    def load_pretrained_weights(self, pretrained_path=''):
        model_dict = self.state_dict()
        staged = ['stage5d', 'stage4d', 'stage3d', 'stage2d', 'stage1d', 'side', 'outconv']

        if not os.path.exists(pretrained_path):
            print(f'\nFile "{pretrained_path}" not exist！')
            exit(1)
        pretrained_dict = torch.load(pretrained_path, map_location={'cuda:0': 'cpu'})
        # pretrained_dict = {k.replace('last_layer', 'aux_head').replace('model.', ''): v for k, v in
        #                    pretrained_dict.items()}
        pretrained_dict_edited = pretrained_dict.copy()
        for key in pretrained_dict.keys():
            if any(stage in key for stage in staged):
                pretrained_dict_edited['segDecoder.'+key] = pretrained_dict_edited[key]
                pretrained_dict_edited['fpmapDecoder.'+key] = pretrained_dict_edited[key]
                pretrained_dict_edited['fnmapDecoder.'+key] = pretrained_dict_edited[key]
                del pretrained_dict_edited[key]

        pretrained_dict_edited = {k: v for k, v in pretrained_dict_edited.items()
                           if k in model_dict.keys()}

        model_dict.update(pretrained_dict_edited)
        self.load_state_dict(model_dict)