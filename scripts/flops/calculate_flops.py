from calflops import calculate_flops
from torchvision import models
from utils import ModifiedResNet, GaussianBlur
from transformers import AutoModelForZeroShotImageClassification

# model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
# model = ModifiedResNet(model)
# batch_size = 1
# input_shape = (batch_size, 3, 224, 224)
# flops, macs, params = calculate_flops(model=model, 
#                                       input_shape=input_shape,
#                                       output_as_string=True,
#                                       output_precision=4)
# print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
# # FLOPs:6.5869 GFLOPS   MACs:3.2779 GMACs   Params:8.5433 M 


# import timm
# import torch
# import os
# local_dir = "/data/checkpoints/UNIv2"
# timm_kwargs = {
#             'model_name': 'vit_giant_patch14_224',
#             'img_size': 224, 
#             'patch_size': 14, 
#             'depth': 24,
#             'num_heads': 24,
#             'init_values': 1e-5, 
#             'embed_dim': 1536,
#             'mlp_ratio': 2.66667*2,
#             'num_classes': 0, 
#             'no_embed_class': True,
#             'mlp_layer': timm.layers.SwiGLUPacked, 
#             'act_layer': torch.nn.SiLU, 
#             'reg_tokens': 8, 
#             'dynamic_img_size': True
#         }
# model = timm.create_model(
#     pretrained=False, **timm_kwargs
# )
# model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
# flops, macs, params = calculate_flops(model=model, 
#                                       input_shape=input_shape,
#                                       output_as_string=True,
#                                       output_precision=4)
# print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
# # FLOPs:360.712 GFLOPS   MACs:180.293 GMACs   Params:681.394 M 


plip = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip")
vision_model = plip.vision_model.eval()            # 只要 Vision Encoder

# 计算CLIP vision model的FLOPS
batch_size = 1
input_shape = (batch_size, 3, 224, 224)  # CLIP通常使用224x224的输入尺寸

# 对于transformer模型，我们需要使用不同的方法
try:
    # 方法1: 不使用input_shape参数
    flops, macs, params = calculate_flops(model=vision_model, 
                                          output_as_string=True,
                                          output_precision=4)
    print("CLIP Vision Model (Method 1):")
    print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
except Exception as e:
    print(f"Method 1 failed: {e}")
    
    # 方法2: 使用transformer_tokenizer参数
    try:
        flops, macs, params = calculate_flops(model=vision_model, 
                                              input_shape=input_shape,
                                              transformer_tokenizer=None,  # 设置为None表示不使用tokenizer
                                              output_as_string=True,
                                              output_precision=4)
        print("CLIP Vision Model (Method 2):")
        print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    except Exception as e:
        print(f"Method 2 failed: {e}")
        
        # 方法3: 使用dummy input
        try:
            import torch
            dummy_input = torch.randn(input_shape)
            flops, macs, params = calculate_flops(model=vision_model, 
                                                  dummy_input=dummy_input,
                                                  output_as_string=True,
                                                  output_precision=4)
            print("CLIP Vision Model (Method 3):")
            print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
        except Exception as e:
            print(f"Method 3 failed: {e}")
            
            # 方法4: 使用thop库
            try:
                import torch
                from thop import profile
                
                dummy_input = torch.randn(input_shape)
                flops, params = profile(vision_model, inputs=(dummy_input,), verbose=False)
                
                # 转换为更易读的格式
                def format_number(num):
                    if num >= 1e9:
                        return f"{num/1e9:.4f} G"
                    elif num >= 1e6:
                        return f"{num/1e6:.4f} M"
                    elif num >= 1e3:
                        return f"{num/1e3:.4f} K"
                    else:
                        return f"{num:.4f}"
                
                print("CLIP Vision Model (Method 4 - thop):")
                print(f"FLOPs: {format_number(flops)}FLOPS")
                print(f"Params: {format_number(params)}")
                print(f"MACs: {format_number(flops/2)}MACs\n")
                
            except ImportError:
                print("thop库未安装，尝试安装...")
                import subprocess
                subprocess.run(["pip", "install", "thop"])
                
                # 重新尝试
                import torch
                from thop import profile
                
                dummy_input = torch.randn(input_shape)
                flops, params = profile(vision_model, inputs=(dummy_input,), verbose=False)
                
                def format_number(num):
                    if num >= 1e9:
                        return f"{num/1e9:.4f} G"
                    elif num >= 1e6:
                        return f"{num/1e6:.4f} M"
                    elif num >= 1e3:
                        return f"{num/1e3:.4f} K"
                    else:
                        return f"{num:.4f}"
                
                print("CLIP Vision Model (Method 4 - thop):")
                print(f"FLOPs: {format_number(flops)}FLOPS")
                print(f"Params: {format_number(params)}")
                print(f"MACs: {format_number(flops/2)}MACs\n")
                
            except Exception as e:
                print(f"Method 4 failed: {e}")
                print("无法计算FLOPS，请检查模型结构或使用其他工具")

# ========== 1. ResNet50 (ModifiedResNet) ==========
try:
    from torchvision import models
    model_r50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    model_r50 = ModifiedResNet(model_r50)
    import torch
    dummy_input = torch.randn(1, 3, 224, 224)
    from thop import profile
    flops, params = profile(model_r50, inputs=(dummy_input,), verbose=False)
    def format_number(num):
        if num >= 1e9:
            return f"{num/1e9:.4f} G"
        elif num >= 1e6:
            return f"{num/1e6:.4f} M"
        elif num >= 1e3:
            return f"{num/1e3:.4f} K"
        else:
            return f"{num:.4f}"
    print("ResNet50 (ModifiedResNet):")
    print(f"FLOPs: {format_number(flops)}FLOPS   MACs: {format_number(flops/2)}MACs   Params: {format_number(params)}\n")
except Exception as e:
    print(f"ResNet50 FLOPs计算失败: {e}")

# ========== 2. UNIv2 (ViT-Giant) ==========
try:
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
    model_uni = timm.create_model(pretrained=False, **timm_kwargs)
    # 权重加载对FLOPS无影响，若缺失权重文件可跳过
    try:
        model_uni.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    except Exception as e:
        print(f"UNIv2权重加载失败: {e}，仅计算结构FLOPS")
    dummy_input = torch.randn(1, 3, 224, 224)
    from thop import profile
    flops, params = profile(model_uni, inputs=(dummy_input,), verbose=False)
    print("UNIv2 (ViT-Giant):")
    print(f"FLOPs: {format_number(flops)}FLOPS   MACs: {format_number(flops/2)}MACs   Params: {format_number(params)}\n")
except Exception as e:
    print(f"UNIv2 FLOPs计算失败: {e}")

# ========== 3. PLIP Vision Encoder (已在下方) ==========
