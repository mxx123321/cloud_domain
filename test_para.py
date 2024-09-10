from return_models import return_models
from ptflops import get_model_complexity_info
import torch
# flops, params = get_model_complexity_info(return_models('SwinUnet'), (3,512,512),as_strings=True,print_per_layer_stat=True)
# print("%s %s" % (flops,params))

# 82.37 G   6.44M   UNet_graph
#236.815G 16.771M   ACC_UNet
#131.490G 28.321M   Rolling_Unet_L
#2.88870G    0.4M   EGEUNet
# 45.396G 41.342M   SwinUnet
#
#
# 

#
# from torchstat import stat

# stat(return_models('SwinUnet'),(3,512,512))

x = torch.randn(1,3,512,512)
from thop import profile
from thop import clever_format
flops, params = profile(return_models('UNext'), inputs=(x, ))
# print(flops, params)
macs, params = clever_format([flops, params], "%.3f")
print(macs,params)
