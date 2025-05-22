from calflops import calculate_flops
from torchvision import models
from utils import ModifiedResNet, GaussianBlur

model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
model = ModifiedResNet(model)
batch_size = 1
input_shape = (batch_size, 3, 224, 224)
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
# FLOPs:6.5869 GFLOPS   MACs:3.2779 GMACs   Params:8.5433 M 


import timm
import torch
import os
local_dir = "/data/checkpoints/UNIv2"
timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
model = timm.create_model(
    pretrained=False, **timm_kwargs
)
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
# FLOPs:360.712 GFLOPS   MACs:180.293 GMACs   Params:681.394 M 