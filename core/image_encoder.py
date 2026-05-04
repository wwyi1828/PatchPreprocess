import os
from typing import Any, Optional, Tuple

import numpy as np
import torch
from PIL import Image


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_pil_image(image: Any) -> Image.Image:
    """Convert arrays/tensors from H5 patches into a PIL image."""
    if isinstance(image, Image.Image):
        return image
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    if arr.dtype != np.uint8:
        max_val = float(arr.max()) if arr.size else 0.0
        if max_val <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        else:
            arr = arr.clip(0, 255)
        arr = arr.astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr))


def _canonicalize_model_type(model_type: str) -> str:
    normalized = model_type.strip().upper()
    alias_map = {
        "VIRCHOW2": "V2",
        "R50": "R50",
        "RESNET50": "RESNET50",
        "RESNET18": "RESNET18",
        "MACENKOUNI": "MACENKOUNI",
    }
    return alias_map.get(normalized, normalized)


def _resolve_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_encoder(
    model_type: str,
    *,
    input_source: str = "h5",
    device: Optional[torch.device] = None,
    uni_ckpt_dir: Optional[str] = None,
) -> Tuple[str, Optional[torch.nn.Module], Any]:
    """
    Load image encoder and processor for a model type.

    input_source:
    - 'h5': processor expects numpy/H5 batch images.
    - 'png': processor expects PIL images from PNGDataset.
    """
    if input_source not in {"h5", "png"}:
        raise ValueError(f"Unsupported input_source: {input_source}")

    canonical_model_type = _canonicalize_model_type(model_type)
    run_device = _resolve_device(device)
    resolved_uni_ckpt_dir = uni_ckpt_dir or os.environ.get("UNI_CKPT_DIR")

    model = None
    image_processor = None

    if canonical_model_type == "PLIP":
        from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

        print("Load PLIP weights")
        model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip")
        image_processor = AutoProcessor.from_pretrained("vinid/plip")
    elif canonical_model_type == "UNI":
        import timm

        if not resolved_uni_ckpt_dir:
            raise ValueError(
                "UNI checkpoint directory is required. Pass --uni_ckpt_dir or set UNI_CKPT_DIR."
            )

        print("Load UNI weights")
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(
            torch.load(os.path.join(resolved_uni_ckpt_dir, "pytorch_model.bin"), map_location="cpu"),
            strict=True,
        )

        if input_source == "h5":
            from transformers import CLIPImageProcessor

            image_processor = CLIPImageProcessor(
                do_resize=False,
                do_center_crop=False,
                do_normalize=True,
                image_mean=_IMAGENET_MEAN,
                image_std=_IMAGENET_STD,
            )
        else:
            from torchvision import transforms

            image_processor = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ]
            )
    elif canonical_model_type == "V2":
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers import SwiGLUPacked

        print("Load Virchow2 weights")
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        image_processor = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    elif canonical_model_type in {"R50", "RESNET50"}:
        from torchvision import models, transforms
        from core.common_utils import ModifiedResNet

        print("Load ResNet50 weights")
        model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        model = ModifiedResNet(model)

        if input_source == "h5":
            from transformers import CLIPImageProcessor

            image_processor = CLIPImageProcessor(
                do_resize=False,
                do_center_crop=False,
                do_normalize=True,
                image_mean=_IMAGENET_MEAN,
                image_std=_IMAGENET_STD,
            )
        else:
            image_processor = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ]
            )
    elif canonical_model_type == "RESNET18":
        from torchvision import models, transforms

        print("Load ResNet18 weights")
        backbone = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        model = torch.nn.Sequential(*list(backbone.children())[:-1])
        image_processor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )
    elif canonical_model_type == "MACENKOUNI":
        import timm
        from torchstain import normalizers as _ts_normalizers
        MacenkoNormalizer = _ts_normalizers.MacenkoNormalizer

        if not resolved_uni_ckpt_dir:
            raise ValueError(
                "UNI checkpoint directory is required for MACENKOUNI. Pass --uni_ckpt_dir or set UNI_CKPT_DIR."
            )

        print("Load MacenkoUNI weights (Macenko stain normalization + UNI)")
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(
            torch.load(os.path.join(resolved_uni_ckpt_dir, "pytorch_model.bin"), map_location="cpu"),
            strict=True,
        )
        # Use built-in default HERef — no fit() needed
        macenko_normalizer = MacenkoNormalizer(backend="torch")
        from transformers import CLIPImageProcessor

        uni_processor = CLIPImageProcessor(
            do_resize=False,
            do_center_crop=False,
            do_normalize=True,
            image_mean=_IMAGENET_MEAN,
            image_std=_IMAGENET_STD,
        )
        image_processor = (macenko_normalizer, uni_processor)
    elif canonical_model_type == "RAW":
        if input_source == "png":
            from torchvision import transforms

            print("Extract Raw images (no encoding)")
            model = torch.nn.Identity()
            image_processor = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ]
            )
    else:
        raise ValueError(f"Unsupported image processor type: {model_type}")

    if model is not None:
        model = model.to(run_device)
        model.eval()

    return canonical_model_type, model, image_processor


def _extract_h5_batch_features(
    batch_images: np.ndarray,
    *,
    model_type: str,
    model: Optional[torch.nn.Module],
    image_processor: Any,
    device: torch.device,
) -> torch.Tensor:
    if model_type == "RAW":
        outputs = torch.as_tensor(batch_images)
        return outputs.unsqueeze(0) if outputs.ndim == 1 else outputs

    if model is None or image_processor is None:
        raise ValueError(f"Model and processor must be loaded for model_type={model_type}")

    if model_type == "MACENKOUNI":
        macenko_normalizer, uni_processor = image_processor
        normalized_images = []
        for img in batch_images:
            # torchstain expects (C, H, W) uint8 tensor
            arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[-1]:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = arr.clip(0, 255).astype(np.uint8)
            t = torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)
            try:
                t_norm, _, _ = macenko_normalizer.normalize(t, stains=True)
                # normalize() returns (H, W, C) int32 — convert back to (H, W, C) uint8
                t_norm = t_norm.clamp(0, 255).to(torch.uint8).numpy()
            except Exception:
                # Fallback: use original image if normalization fails (e.g. near-white patch)
                t_norm = arr
            normalized_images.append(t_norm)
        inputs = uni_processor(images=normalized_images, return_tensors="pt")["pixel_values"]
        if inputs.ndim == 5 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        outputs = model(inputs.to(device))
        if torch.is_tensor(outputs) and outputs.ndim > 2:
            outputs = outputs[:, 0, :]
    elif model_type == "V2":
        processed = [image_processor(ensure_pil_image(img)) for img in batch_images]
        inputs = torch.stack(processed).to(device)
        outputs = model(inputs)
        outputs = outputs[:, 0, :]
    elif model_type == "PLIP":
        inputs = image_processor(images=list(batch_images), return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        if pixel_values.ndim == 5 and pixel_values.shape[1] == 1:
            pixel_values = pixel_values.squeeze(1)
        outputs = model.vision_model(pixel_values=pixel_values.to(device)).pooler_output
    else:
        inputs = image_processor(images=list(batch_images), return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        if pixel_values.ndim == 5 and pixel_values.shape[1] == 1:
            pixel_values = pixel_values.squeeze(1)
        outputs = model(pixel_values.to(device))
        if hasattr(outputs, "pooler_output"):
            outputs = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            outputs = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if torch.is_tensor(outputs) and outputs.ndim > 2:
            outputs = outputs[:, 0, :]

    if not torch.is_tensor(outputs):
        outputs = torch.as_tensor(outputs)
    return outputs.unsqueeze(0) if outputs.ndim == 1 else outputs


def extract_h5_features_in_batches(
    images: np.ndarray,
    *,
    model_type: str,
    model: Optional[torch.nn.Module],
    image_processor: Any,
    batch_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Run H5 morphology images through encoder and return concatenated features."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    run_device = _resolve_device(device)
    canonical_model_type = _canonicalize_model_type(model_type)

    num_images = int(images.shape[0])
    features = []
    with torch.no_grad():
        for start in range(0, num_images, batch_size):
            end = min(start + batch_size, num_images)
            batch = images[start:end]
            batch_features = _extract_h5_batch_features(
                batch,
                model_type=canonical_model_type,
                model=model,
                image_processor=image_processor,
                device=run_device,
            )
            features.append(batch_features.cpu())

    if not features:
        return torch.empty((0,), dtype=torch.float32)
    return torch.cat(features, dim=0)
