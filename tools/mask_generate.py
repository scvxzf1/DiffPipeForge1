
import os
import argparse
import logging
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Segformer B2 Clothes label mapping
# 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes",
# 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe",
# 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
# 16: "Bag", 17: "Scarf"

LABEL_MAP = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglass": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "face": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

# Friendly label names that UI sends (shoe maps to both left_shoe and right_shoe)
LABEL_ALIASES = {
    "shoe": [9, 10],
}


def get_label_ids(label_names):
    """Convert label name strings to label IDs."""
    ids = []
    for name in label_names:
        name = name.strip().lower()
        if name in LABEL_ALIASES:
            ids.extend(LABEL_ALIASES[name])
        elif name in LABEL_MAP:
            ids.append(LABEL_MAP[name])
        else:
            logger.warning(f"Unknown label: {name}")
    return ids


def setup_model(model_path):
    """Load Segformer B2 Clothes model."""
    import torch
    from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info(f"Loading Segformer from {model_path}...")
    processor = SegformerImageProcessor.from_pretrained(model_path)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    logger.info(f"Model loaded on {device}")
    return processor, model, device


def get_segmentation(image, processor, model, device):
    """Run segmentation inference on a PIL image."""
    import torch
    import torch.nn as nn

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.cpu()
    upsampled = nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred_seg = upsampled.argmax(dim=1)[0].numpy()
    return pred_seg


# ============================================================
# Edge Detail Processing (standalone, no ComfyUI dependency)
# ============================================================

def guided_filter(guide, src, radius, eps=1e-6):
    """Simple guided filter implementation using box filter (PIL blur)."""

    def box_filter(img, r):
        return img.filter(ImageFilter.BoxBlur(r))

    guide_np = np.array(guide, dtype=np.float32) / 255.0
    src_np = np.array(src, dtype=np.float32) / 255.0

    if guide_np.ndim == 3:
        guide_gray = np.mean(guide_np, axis=2)
    else:
        guide_gray = guide_np

    guide_pil = Image.fromarray((guide_gray * 255).astype(np.uint8), 'L')
    src_pil = Image.fromarray((src_np * 255).astype(np.uint8), 'L')

    mean_I = np.array(box_filter(guide_pil, radius), dtype=np.float32) / 255.0
    mean_p = np.array(box_filter(src_pil, radius), dtype=np.float32) / 255.0

    Ip = Image.fromarray(((guide_gray * src_np) * 255).astype(np.uint8), 'L')
    mean_Ip = np.array(box_filter(Ip, radius), dtype=np.float32) / 255.0
    cov_Ip = mean_Ip - mean_I * mean_p

    II = Image.fromarray(((guide_gray * guide_gray) * 255).astype(np.uint8), 'L')
    mean_II = np.array(box_filter(II, radius), dtype=np.float32) / 255.0
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    a_pil = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), 'L')
    b_pil = Image.fromarray((np.clip(b, 0, 1) * 255).astype(np.uint8), 'L')

    mean_a = np.array(box_filter(a_pil, radius), dtype=np.float32) / 255.0
    mean_b = np.array(box_filter(b_pil, radius), dtype=np.float32) / 255.0

    result = mean_a * guide_gray + mean_b
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result, 'L')


def histogram_remap(mask_image, black_point, white_point):
    """Remap mask histogram to expand contrast."""
    mask_np = np.array(mask_image, dtype=np.float32) / 255.0
    remapped = (mask_np - black_point) / max(white_point - black_point, 1e-6)
    remapped = np.clip(remapped, 0, 1)
    return Image.fromarray((remapped * 255).astype(np.uint8), 'L')


def process_detail_guided_filter(orig_image, mask_image, detail_range, black_point, white_point):
    """Edge refinement using guided filter."""
    radius = detail_range // 6 + 1
    refined = guided_filter(orig_image, mask_image, radius)
    return histogram_remap(refined, black_point, white_point)


def process_detail_pymatting(orig_image, mask_image, detail_range, black_point, white_point):
    """Edge refinement using PyMatting."""
    try:
        from pymatting import estimate_alpha_cf
    except ImportError:
        logger.warning("pymatting not installed, falling back to GuidedFilter")
        return process_detail_guided_filter(orig_image, mask_image, detail_range, black_point, white_point)

    orig_np = np.array(orig_image.convert('RGB'), dtype=np.float64) / 255.0
    mask_np = np.array(mask_image, dtype=np.float64) / 255.0

    # Create trimap from mask
    erode_size = detail_range // 8 + 1
    from PIL import ImageFilter
    eroded = mask_image.filter(ImageFilter.MinFilter(erode_size * 2 + 1))
    dilated = mask_image.filter(ImageFilter.MaxFilter(erode_size * 2 + 1))

    trimap = np.array(mask_image, dtype=np.float64) / 255.0
    eroded_np = np.array(eroded, dtype=np.float64) / 255.0
    dilated_np = np.array(dilated, dtype=np.float64) / 255.0

    # Unknown region between eroded and dilated
    trimap = np.where(eroded_np > 0.5, 1.0, np.where(dilated_np < 0.5, 0.0, 0.5))

    alpha = estimate_alpha_cf(orig_np, trimap)
    alpha = np.clip(alpha, 0, 1)
    result = Image.fromarray((alpha * 255).astype(np.uint8), 'L')
    return histogram_remap(result, black_point, white_point)


def process_detail_vitmatte(orig_image, mask_image, detail_erode, detail_dilate,
                             black_point, white_point, model_path, local_files_only=False,
                             device='cpu', max_megapixels=2.0):
    """Edge refinement using VITMatte model."""
    try:
        import torch
        from transformers import VitMatteImageProcessor, VitMatteForImageMatting
    except ImportError:
        logger.warning("VITMatte not available, falling back to GuidedFilter")
        detail_range = detail_erode + detail_dilate
        return process_detail_guided_filter(orig_image, mask_image, detail_range, black_point, white_point)

    vitmatte_path = os.path.join(os.path.dirname(model_path), 'vitmatte-small-composition-1k')
    if not os.path.exists(vitmatte_path):
        logger.warning(f"VITMatte model not found at {vitmatte_path}, falling back to GuidedFilter")
        detail_range = detail_erode + detail_dilate
        return process_detail_guided_filter(orig_image, mask_image, detail_range, black_point, white_point)

    # Resize if too large
    w, h = orig_image.size
    megapixels = (w * h) / 1e6
    if megapixels > max_megapixels:
        scale = (max_megapixels / megapixels) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        orig_resized = orig_image.resize((new_w, new_h), Image.LANCZOS)
        mask_resized = mask_image.resize((new_w, new_h), Image.LANCZOS)
    else:
        orig_resized = orig_image
        mask_resized = mask_image

    # Generate trimap from mask
    eroded = mask_resized.filter(ImageFilter.MinFilter(detail_erode * 2 + 1))
    dilated = mask_resized.filter(ImageFilter.MaxFilter(detail_dilate * 2 + 1))
    eroded_np = np.array(eroded, dtype=np.float64) / 255.0
    dilated_np = np.array(dilated, dtype=np.float64) / 255.0
    trimap_np = np.where(eroded_np > 0.5, 255, np.where(dilated_np < 0.5, 0, 128)).astype(np.uint8)
    trimap_pil = Image.fromarray(trimap_np, 'L')

    # Load VITMatte
    vit_processor = VitMatteImageProcessor.from_pretrained(vitmatte_path, local_files_only=local_files_only)
    vit_model = VitMatteForImageMatting.from_pretrained(vitmatte_path, local_files_only=local_files_only)
    vit_model.to(device)
    vit_model.eval()

    inputs = vit_processor(images=orig_resized.convert('RGB'), trimaps=trimap_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        alphas = vit_model(**inputs).alphas

    alpha = alphas[0, 0].cpu().numpy()
    alpha = np.clip(alpha, 0, 1)
    alpha_pil = Image.fromarray((alpha * 255).astype(np.uint8), 'L')

    # Resize back if needed
    if megapixels > max_megapixels:
        alpha_pil = alpha_pil.resize((w, h), Image.LANCZOS)

    return histogram_remap(alpha_pil, black_point, white_point)


def process_images(input_dir, output_dir, processor, model, device,
                   selected_labels, detail_method='GuidedFilter',
                   detail_erode=12, detail_dilate=6,
                   black_point=0.15, white_point=0.99,
                   process_detail=True, max_megapixels=2.0,
                   model_path='', output_white_level=1.0, output_black_level=0.0):
    """Process all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

    if not files:
        logger.warning(f"No images found in {input_dir}")
        return

    logger.info(f"Found {len(files)} images. Selected labels: {selected_labels}")
    logger.info(f"Detail method: {detail_method}, Process detail: {process_detail}")

    # Build labels_to_keep (labels NOT selected = background)
    # The logic: selected labels become foreground (white), everything else is background (black)
    selected_ids = get_label_ids(selected_labels) if selected_labels else []
    # labels_to_keep are the ones we want to mark as background (black) in the final mask
    # i.e., everything that is NOT selected
    all_ids = list(range(18))
    labels_to_keep = [lid for lid in all_ids if lid not in selected_ids]

    detail_range = detail_erode + detail_dilate

    for file_path in tqdm(files, desc="Processing Images"):
        try:
            img = Image.open(file_path).convert("RGB")
            pred_seg = get_segmentation(img, processor, model, device)

            # Create mask: selected parts = white, rest = black
            mask = np.isin(pred_seg, labels_to_keep).astype(np.uint8)
            mask_image = Image.fromarray((1 - mask) * 255).convert("L")

            # Edge detail processing
            if process_detail and detail_method != 'None':
                if detail_method == 'GuidedFilter':
                    mask_image = process_detail_guided_filter(
                        img, mask_image, detail_range, black_point, white_point
                    )
                elif detail_method == 'PyMatting':
                    mask_image = process_detail_pymatting(
                        img, mask_image, detail_range, black_point, white_point
                    )
                    mask_image = process_detail_vitmatte(
                        img, mask_image, detail_erode, detail_dilate,
                        black_point, white_point, model_path,
                        local_files_only=True, device=device,
                        max_megapixels=max_megapixels
                    )

            # Output level adjustment
            if output_white_level != 1.0 or output_black_level != 0.0:
                mask_np = np.array(mask_image, dtype=np.float32) / 255.0
                # Formula: final = pixel * (white - black) + black
                # This ensures 1.0 -> white_level, 0.0 -> black_level
                mask_np = mask_np * (output_white_level - output_black_level) + output_black_level
                mask_np = np.clip(mask_np, 0, 1)
                mask_image = Image.fromarray((mask_np * 255).astype(np.uint8), 'L')

            save_path = output_path / f"{file_path.stem}.png"
            mask_image.save(save_path)
            logger.info(f"Saved mask: {save_path}")

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")

    logger.info("Processing finished.")


def download_model(model_path, source="modelscope", model_type="segformer"):
    """Download model to specific directory."""
    REPO_MAP = {
        "segformer": {
            "modelscope": "Tiandong/segformer_b2_clothes",
            "huggingface": "mattmdjaga/segformer_b2_clothes",
        },
        "vitmatte": {
            "modelscope": "Tiandong/vitmatte-small-composition-1k",
            "huggingface": "hustvl/vitmatte-small-composition-1k",
        },
    }

    repo_info = REPO_MAP.get(model_type, REPO_MAP["segformer"])
    repo_id = repo_info.get(source, repo_info["modelscope"])

    logger.info(f"Downloading {model_type} model ({repo_id}) to {model_path} using {source}...")
    try:
        os.makedirs(model_path, exist_ok=True)
        if source == "modelscope":
            from modelscope.hub.snapshot_download import snapshot_download
            snapshot_download(repo_id, local_dir=model_path)
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo_id, local_dir=model_path)

        logger.info(f"{model_type} model download completed successfully.")
        return True
    except ImportError as e:
        logger.error(f"Package not found: {e}")
        print(f"Error: Required package for {source} not found.")
        return False
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Segformer B2 Clothes Mask Generator")
    parser.add_argument("--mode", type=str, choices=["check", "download", "run"], required=True)
    parser.add_argument("--input_dir", type=str, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, help="Directory to save masks")
    parser.add_argument("--model_path", type=str, default="segformer/segformer_b2_clothes",
                        help="Path to model directory")
    parser.add_argument("--source", type=str, choices=["huggingface", "modelscope"], default="modelscope")
    parser.add_argument("--model_type", type=str, choices=["segformer", "vitmatte", "all"], default="segformer")
    parser.add_argument("--online", action="store_true", help="Allow online access")

    # Segmentation labels
    parser.add_argument("--labels", type=str, default="",
                        help="Comma-separated label names to include in mask (e.g., face,hair,upper_clothes)")

    # Detail processing
    parser.add_argument("--detail_method", type=str, default="VITMatte",
                        choices=["GuidedFilter", "PyMatting", "VITMatte", "None"])
    parser.add_argument("--detail_erode", type=int, default=12)
    parser.add_argument("--detail_dilate", type=int, default=6)
    parser.add_argument("--black_point", type=float, default=0.15)
    parser.add_argument("--white_point", type=float, default=0.99)
    parser.add_argument("--process_detail", type=str, default="true",
                        help="Enable detail processing (true/false)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max_megapixels", type=float, default=2.0)
    parser.add_argument("--output_white_level", type=float, default=1.0, help="Output white level (0.0-1.0)")
    parser.add_argument("--output_black_level", type=float, default=0.0, help="Output black level (0.0-1.0)")

    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Resolve model path
    if os.path.isabs(args.model_path):
        model_abs_path = Path(args.model_path)
    else:
        model_abs_path = script_dir / args.model_path

    # VITMatte path is always sibling to Segformer path
    vitmatte_abs_path = model_abs_path.parent / "vitmatte-small-composition-1k"

    if args.mode == "check":
        # Check both Segformer and VITMatte
        seg_exists = (model_abs_path / "config.json").exists() and \
                     ((model_abs_path / "model.safetensors").exists() or
                      (model_abs_path / "pytorch_model.bin").exists())
        
        vit_exists = (vitmatte_abs_path / "config.json").exists() and \
                     ((vitmatte_abs_path / "model.safetensors").exists() or
                      (vitmatte_abs_path / "pytorch_model.bin").exists())

        is_exists = seg_exists and vit_exists
        print(f"MODEL_EXISTS: {is_exists}")
        if not is_exists:
            logger.info(f"Model check failed. Segformer: {seg_exists}, VITMatte: {vit_exists}")
        return

    if args.mode == "download":
        success = True
        
        if args.model_type == "all" or args.model_type == "segformer":
            logger.info("Downloading Segformer...")
            if not download_model(model_abs_path, source=args.source, model_type="segformer"):
                success = False

        if args.model_type == "all" or args.model_type == "vitmatte":
            logger.info("Downloading VITMatte...")
            if not download_model(vitmatte_abs_path, source=args.source, model_type="vitmatte"):
                success = False

        if success:
            print("DOWNLOAD_SUCCESS")
        else:
            print("DOWNLOAD_FAILED")
            sys.exit(1)
        return

    if args.mode == "run":
        if not args.input_dir:
            logger.error("Input directory is required for run mode")
            sys.exit(1)

        if not args.output_dir:
            args.output_dir = args.input_dir.rstrip('/\\') + "_masks"

        # Validate: mask output dir must not be inside input dir
        abs_input = os.path.abspath(args.input_dir)
        abs_output = os.path.abspath(args.output_dir)
        if abs_output.startswith(abs_input + os.sep):
            logger.error(f"Output directory ({abs_output}) must not be inside input directory ({abs_input})")
            print("ERROR: Mask output directory cannot be a subfolder of the input directory.")
            sys.exit(1)

        # Parse labels
        selected_labels = [l.strip() for l in args.labels.split(',') if l.strip()] if args.labels else []

        if not selected_labels:
            logger.error("No labels selected. Please select at least one body part.")
            print("ERROR: No labels selected.")
            sys.exit(1)

        # Resolve device
        import torch
        if args.device == "auto":
            device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_to_use = args.device

        processor, model, _ = setup_model(model_abs_path)
        # Override device
        model.to(device_to_use)

        process_detail = args.process_detail.lower() == 'true'

        process_images(
            args.input_dir, args.output_dir, processor, model, device_to_use,
            selected_labels,
            detail_method=args.detail_method,
            detail_erode=args.detail_erode,
            detail_dilate=args.detail_dilate,
            black_point=args.black_point,
            white_point=args.white_point,
            process_detail=process_detail,
            max_megapixels=args.max_megapixels,
            model_path=str(model_abs_path),
            output_white_level=args.output_white_level,
            output_black_level=args.output_black_level,
        )


if __name__ == "__main__":
    main()
