import os
import sys

import torch


def format_number(num: float) -> str:
    if num >= 1e12:
        return f"{num/1e12:.4f} T"
    if num >= 1e9:
        return f"{num/1e9:.4f} G"
    if num >= 1e6:
        return f"{num/1e6:.4f} M"
    if num >= 1e3:
        return f"{num/1e3:.4f} K"
    return f"{num:.4f}"


def build_uni_model():
    try:
        import timm
    except Exception as e:
        print(f"timm 未安装或导入失败: {e}")
        sys.exit(1)

    # UNI uses ViT-L/16 with these args in this repo
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )

    # Optional: load weights (not required for FLOPs)
    ckpt_dir = os.environ.get("UNI_CKPT_DIR", "/data/checkpoints/UNI")
    ckpt_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.isfile(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
            print(f"已加载权重: {ckpt_path}")
        except Exception as e:
            print(f"权重加载失败(忽略以计算FLOPs): {e}")
    else:
        print(f"未找到权重 {ckpt_path}，仅计算结构FLOPs")

    model.eval()
    return model


def main():
    try:
        from thop import profile
    except Exception as e:
        print(f"thop 未安装或导入失败: {e}")
        sys.exit(1)

    model = build_uni_model()
    dummy = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    print("UNI (ViT-L/16, 224x224):")
    print(f"FLOPs: {format_number(flops)}FLOPS")
    print(f"MACs:  {format_number(flops/2)}MACs")
    print(f"Params:{format_number(params)}")


if __name__ == "__main__":
    main()

