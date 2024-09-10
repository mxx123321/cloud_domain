from model.ICIFNet import ICIFNet
import torch
from models.unet.unet_model import UNet#(10,10).cuda()


# model = ICIFNet(pretrained=False)

# tensor = torch.randn(3,3,512,512)

# output = model(tensor,tensor)
# print("output",output[0].shape)


model = UNet(3,2)
tensor = torch.randn(2,3,512,512)

print(model(tensor).shape)