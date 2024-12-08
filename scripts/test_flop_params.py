import torch
from thop import profile
from thop import clever_format
from isegm.inference import utils

input = torch.randn(1, 6, 350, 740)
point = torch.randn(1, 2, 3)
pseudo_points = torch.randint(0,1,size=(1, 66, 3))
model = utils.load_is_model('../weights/test.pth', 'cpu')

macs, params = profile(model, inputs=(input, point, pseudo_points))
gflops, params = clever_format([macs*2, params], "%.5f")

print(gflops, params)