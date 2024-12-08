import os

import torch
import torch.nn as nn
import torch.nn.functional as F

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



class Attention_block(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


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

# =============================================seg decoder=================================================================
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

# =============================================error prediction decoder（fpmap）======================================================
class FPMAPDecoder(nn.Module):

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

            # self.att5 = Attention_block(F_g=64, F_l=64, F_int=16)
            # self.att4 = Attention_block(F_g=64, F_l=64, F_int=16)
            # self.att3 = Attention_block(F_g=64, F_l=64, F_int=16)
            # self.att2 = Attention_block(F_g=64, F_l=64, F_int=16)
            # self.att1 = Attention_block(F_g=64, F_l=64, F_int=16)

            self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, hx1, hx2, hx3, hx4, hx5, hx6):

        hx6up = _upsample_like(hx6, hx5)
        # hx5 = self.att5(g=hx6up, x=hx5)

        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))

        hx5dup = _upsample_like(hx5d, hx4)
        # hx4 = self.att4(g=hx5dup, x=hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))

        hx4dup = _upsample_like(hx4d, hx3)
        # hx3 = self.att3(g=hx4dup, x=hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))

        hx3dup = _upsample_like(hx3d, hx2)
        # hx2 = self.att2(g=hx3dup, x=hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))

        hx2dup = _upsample_like(hx2d, hx1)
        # hx1 = self.att1(g=hx2dup, x=hx1)

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

            # self.att5 = Attention_block(F_g=64, F_l=64, F_int=16)
            # self.att4 = Attention_block(F_g=64, F_l=64, F_int=16)
            # self.att3 = Attention_block(F_g=64, F_l=64, F_int=16)
            # self.att2 = Attention_block(F_g=64, F_l=64, F_int=16)
            # self.att1 = Attention_block(F_g=64, F_l=64, F_int=16)

            self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, hx1, hx2, hx3, hx4, hx5, hx6):

        hx6up = _upsample_like(hx6, hx5)
        # hx5 = self.att5(g=hx6up, x=hx5)

        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))

        hx5dup = _upsample_like(hx5d, hx4)
        # hx4 = self.att4(g=hx5dup, x=hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))

        hx4dup = _upsample_like(hx4d, hx3)
        # hx3 = self.att3(g=hx4dup, x=hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))

        hx3dup = _upsample_like(hx3d, hx2)
        # hx2 = self.att2(g=hx3dup, x=hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))

        hx2dup = _upsample_like(hx2d, hx1)
        # hx1 = self.att1(g=hx2dup, x=hx1)

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
class PseudoScribbleU2NETP(nn.Module):

    def __init__(self,in_ch=3, out_ch=1):
        super(PseudoScribbleU2NETP,self).__init__()

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
        fp_fn_coord_features  = None
        fp_coord_features = fn_coord_features = None
        if coord_features.size()[1] > 64:
            # Not introducing prior masks for false positives and false negatives
            fp_fn_coord_features = coord_features[:, :64, :, :]
            coord_features = coord_features[:, 64:, :, :]

        hx = x
        fp_fn_hx = x.clone()
        fp_hx = x.clone()
        fn_hx = x.clone()
        #================================================encoding=================================================
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

        fp_fn_hx1 = hx1
        fp_fn_hx2 = hx2
        fp_fn_hx3 = hx3
        fp_fn_hx4 = hx4
        fp_fn_hx5 = hx5
        fp_fn_hx6 = hx6


        if fp_fn_coord_features is not None:
            # stage 1
            fp_fn_hx1 = self.stage1(fp_fn_hx, fp_fn_coord_features)
            fp_fn_hx = self.pool12(fp_fn_hx1)

            # stage 2
            fp_fn_hx2 = self.stage2(fp_fn_hx)
            fp_fn_hx = self.pool23(fp_fn_hx2)

            # stage 3
            fp_fn_hx3 = self.stage3(fp_fn_hx)
            fp_fn_hx = self.pool34(fp_fn_hx3)

            # stage 4
            fp_fn_hx4 = self.stage4(fp_fn_hx)
            fp_fn_hx = self.pool45(fp_fn_hx4)

            # stage 5
            fp_fn_hx5 = self.stage5(fp_fn_hx)
            fp_fn_hx = self.pool56(fp_fn_hx5)

            # stage 6
            fp_fn_hx6 = self.stage6(fp_fn_hx)

        # 解码阶段
        seg_output, seg_side_output = self.segDecoder(hx1, hx2, hx3, hx4, hx5, hx6)
        if fp_fn_coord_features is None:
            return [seg_output, seg_side_output, None, None, None, None]
        fp_output, fp_side_output = self.fpmapDecoder(fp_fn_hx1, fp_fn_hx2, fp_fn_hx3, fp_fn_hx4, fp_fn_hx5, fp_fn_hx6)
        fn_output, fn_side_output = self.fnmapDecoder(fp_fn_hx1, fp_fn_hx2, fp_fn_hx3, fp_fn_hx4, fp_fn_hx5, fp_fn_hx6)
        # fp_output, fp_side_output = self.fpmapDecoder(fp_hx1, fp_hx2, fp_hx3, fp_hx4, fp_hx5, fp_hx6)
        # fp_output, fp_side_output = self.fpmapDecoder(hx1, hx2, hx3, hx4, hx5, hx6)
        # fn_output, fn_side_output = self.fnmapDecoder(fn_hx1, fn_hx2, fn_hx3, fn_hx4, fn_hx5, fn_hx6)
        # fn_output, fn_side_output = self.fnmapDecoder(hx1, hx2, hx3, hx4, hx5, hx6)
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