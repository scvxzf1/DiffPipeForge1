import os
import shutil
import argparse
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

r"""
风格筛选脚本 (filter_style.py)

功能：
    使用 CLIP 模型根据文本提示词筛选图片风格。
    主要用于区分“二次元/动漫”与“写实/照片”风格，或者其他自定义风格。

依赖：
    pip install torch transformers pillow

使用方法：
    1. 默认运行 (保留二次元，移走写实):
       python filter_style.py

    2. 自定义提示词:
       python filter_style.py --keep "anime style, 2d art" --remove "realistic photo, 3d render"

    3. 指定模型和批量大小 (4090 推荐大批量):
       python filter_style.py --batch-size 64 --model "C:\Users\LUYUE\Desktop\1\clip-vit-base-patch32"

    4. 指定线程数 (加快图片读取):
       python filter_style.py --threads 16

    5.如：
     python filter_style.py --dir "C:\Users\LUYUE\Desktop\1" --keep "anime style, 2d art" --threads 16 --remove "man,boy,animal,subjcet" --batch-size 128 --model "D:\工具\clip模型处理\clip-vit-base-patch32"

参数说明：
    --dir:        扫描目录 (默认: 当前目录)
    --keep:       保留风格的提示词，用逗号分隔 (默认: "anime, 2d illustration, drawing")
    --remove:     移除风格的提示词，用逗号分隔 (默认: "photorealistic, realistic, photo, 3d render")
    --batch-size: 批处理大小。显存越大可以设越大 (默认: 32)
    --model:      HuggingFace 模型名称或本地路径 (默认: openai/clip-vit-base-patch32)
    --threads:    读取图片的线程数 (默认: 8)
"""

def is_image_file(filename):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    return os.path.splitext(filename)[1].lower() in image_extensions

def move_to_folder(file_path, root_dir, folder_name):
    target_dir = os.path.join(root_dir, folder_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        
    dest_path = os.path.join(target_dir, os.path.basename(file_path))
    
    if os.path.exists(dest_path):
        base, ext = os.path.splitext(os.path.basename(file_path))
        counter = 1
        while os.path.exists(os.path.join(target_dir, f"{base}_{counter}{ext}")):
            counter += 1
        dest_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
        
    try:
        shutil.move(file_path, dest_path)
        return True
    except Exception as e:
        print(f"移动失败 {file_path}: {e}")
        return False

def load_image(file_path):
    """读取并转换图片为 RGB"""
    try:
        image = Image.open(file_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return file_path, image
    except Exception as e:
        print(f"无法读取 {os.path.basename(file_path)}: {e}")
        return file_path, None

def main():
    parser = argparse.ArgumentParser(description="使用 CLIP 进行筛选。", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dir", type=str, default=".", help="要扫描的目录")
    parser.add_argument("--keep", type=str, default="anime, 2d illustration, drawing, flat color", help="保留的风格提示词 (逗号分隔)")
    parser.add_argument("--remove", type=str, default="photorealistic, realistic, photo, 3d render, real person", help="移除的风格提示词 (逗号分隔)")
    parser.add_argument("--batch-size", type=int, default=32, help="批处理大小 (默认: 32)")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", help="模型名称或路径")
    parser.add_argument("--threads", type=int, default=8, help="读取图片的线程数 (默认: 8)")
    
    args = parser.parse_args()
    
    # 检查 CUDA
    print(f"Torch 版本: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"显卡型号: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("警告: Torch 未检测到 CUDA。这通常是因为安装了 CPU 版本的 PyTorch。")
        device = "cpu"
    
    print(f"运行设备: {device.upper()}")
    
    # 加载模型
    print(f"正在加载模型: {args.model} ...")
    try:
        model = CLIPModel.from_pretrained(args.model).to(device)
        processor = CLIPProcessor.from_pretrained(args.model)
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请检查网络连接或模型路径。")
        return

    # 准备提示词
    keep_texts = [t.strip() for t in args.keep.split(",")]
    remove_texts = [t.strip() for t in args.remove.split(",")]
    all_texts = keep_texts + remove_texts
    
    print(f"保留: {keep_texts}")
    print(f"移除: {remove_texts}")
    
    # 扫描文件
    directory = os.path.abspath(args.dir)
    print(f"扫描目录: {directory}")
    
    image_files = []
    for root, _, files in os.walk(directory):
        if "style_mismatch" in root or "duplicates" in root or "low_quality" in root:
            continue
        for file in files:
            if is_image_file(file):
                image_files.append(os.path.join(root, file))
                
    print(f"找到 {len(image_files)} 张图片。")
    
    # 批处理
    moved_count = 0
    
    # 将文件列表分批
    batches = [image_files[i:i + args.batch_size] for i in range(0, len(image_files), args.batch_size)]
    
    print(f"开始处理... (使用 {args.threads} 线程加载图片)")
    
    for batch in tqdm(batches, unit="batch"):
        images = []
        valid_files = []
        
        # 并行加载图片
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            results = list(executor.map(load_image, batch))
            
        for file_path, img in results:
            if img is not None:
                images.append(img)
                valid_files.append(file_path)
        
        if not images:
            continue
            
        # 推理
        try:
            inputs = processor(text=all_texts, images=images, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image # [batch_size, num_text]
                probs = logits_per_image.softmax(dim=1) # 归一化概率
                
            # 分析结果
            keep_indices = list(range(len(keep_texts)))
            remove_indices = list(range(len(keep_texts), len(all_texts)))
            
            probs_cpu = probs.cpu().numpy()
            
            for i, file_path in enumerate(valid_files):
                keep_score = probs_cpu[i][keep_indices].sum()
                remove_score = probs_cpu[i][remove_indices].sum()
                
                if remove_score > keep_score:
                    # 判定为移除风格
                    if move_to_folder(file_path, directory, "style_mismatch"):
                        moved_count += 1
                        print(f"  [移出] {os.path.basename(file_path)} (移除分: {remove_score:.2f} > 保留分: {keep_score:.2f})")
                        
        except Exception as e:
            print(f"批处理出错: {e}")
            
    print("\n筛选完成！")
    print(f"共移出不符合的图片: {moved_count}")
    print(f"存放位置: {os.path.join(directory, 'style_mismatch')}")

if __name__ == "__main__":
    main()
