import os
import shutil
import argparse
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

"""
图片质量筛选脚本 (filter_quality.py)

功能：
    扫描指定目录，筛选出质量不合格的图片（分辨率低、文件小、模糊、噪声大、对比度异常等），
    并将它们移动到 `low_quality` 文件夹中。

使用方法：
    1. 默认运行 (筛选分辨率<512x512, 大小<50KB):
       python filter_quality.py

    2. 自定义所有阈值:
       python filter_quality.py --dir "C:/Images" \
           --min-width 1024 --min-height 1024 \
           --min-size 100 \
           --blur-threshold 100 \
           --max-noise 50 \
           --min-contrast 20 --max-contrast 100
           --threads 4
    3.如：
        python filter_quality.py --min-size 100 --blur-threshold 100 --max-noise 50 --min-contrast 20 --max-contrast 100 --threads 4

参数说明：
    --dir:            扫描目录 (默认: 当前目录)
    --min-width:      最小宽度 (默认: 512)
    --min-height:     最小高度 (默认: 512)
    --min-size:       最小文件大小 (KB, 默认: 50)
    --blur-threshold: 模糊度阈值 (拉普拉斯方差)。低于此值视为模糊 (默认: 0, 不检测。建议: 100-300)
    --max-noise:      最大噪声阈值 (估计值)。高于此值视为噪点过多 (默认: 0, 不检测。建议: 20-50)
    --min-contrast:   最小对比度 (像素标准差)。低于此值视为灰蒙蒙/无细节 (默认: 0, 不检测。建议: 10-20)
    --max-contrast:   最大对比度 (像素标准差)。高于此值视为异常 (默认: 0, 不检测)
    --threads:        线程数 (默认: 4)
"""

def is_image_file(filename):
    """检查文件是否为图片。"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    return os.path.splitext(filename)[1].lower() in image_extensions

def estimate_noise(img_gray):
    """
    估计图像噪声水平。
    使用简单的算法：计算图像与中值滤波后的差值的标准差。
    """
    # 使用 3x3 中值滤波去除噪声
    img_median = cv2.medianBlur(img_gray, 3)
    # 计算差值
    diff = cv2.absdiff(img_gray, img_median)
    # 差值的平均值或标准差可以作为噪声的估计
    # 这里使用平均值，简单且对异常值不敏感
    return np.mean(diff)

def check_image_quality(file_path, args):
    """
    检查图片质量。
    返回: (是否通过, 失败原因)
    """
    try:
        # 1. 检查文件大小
        file_size_kb = os.path.getsize(file_path) / 1024
        if file_size_kb < args.min_size:
            return False, f"文件过小 ({file_size_kb:.1f}KB < {args.min_size}KB)"

        # 2. 检查分辨率 (使用 PIL 读取，速度快)
        with Image.open(file_path) as img:
            width, height = img.size
            if width < args.min_width or height < args.min_height:
                return False, f"分辨率过低 ({width}x{height} < {args.min_width}x{args.min_height})"
        
        # 需要读取像素数据的检查 (模糊、噪声、对比度)
        if args.blur_threshold > 0 or args.max_noise > 0 or args.min_contrast > 0 or args.max_contrast > 0:
            # cv2.imread 不支持中文路径，需要用 imdecode
            img_array = np.fromfile(file_path, np.uint8)
            img_gray = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            if img_gray is None:
                return False, "无法读取图像数据"
            
            # 3. 检查模糊度 (拉普拉斯方差)
            if args.blur_threshold > 0:
                laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
                if laplacian_var < args.blur_threshold:
                    return False, f"图像模糊 (清晰度 {laplacian_var:.1f} < {args.blur_threshold})"

            # 4. 检查噪声 (Noise)
            if args.max_noise > 0:
                noise_level = estimate_noise(img_gray)
                if noise_level > args.max_noise:
                    return False, f"噪声过大 (噪声值 {noise_level:.1f} > {args.max_noise})"

            # 5. 检查对比度/像素差 (Contrast / Pixel Variation)
            # 使用像素标准差作为对比度的简单度量
            if args.min_contrast > 0 or args.max_contrast > 0:
                std_dev = np.std(img_gray)
                if args.min_contrast > 0 and std_dev < args.min_contrast:
                    return False, f"对比度过低 (标准差 {std_dev:.1f} < {args.min_contrast})"
                if args.max_contrast > 0 and std_dev > args.max_contrast:
                    return False, f"对比度异常高 (标准差 {std_dev:.1f} > {args.max_contrast})"

        return True, "合格"

    except Exception as e:
        return False, f"处理出错: {e}"

def move_to_low_quality(file_path, root_dir, reason):
    """将不合格图片移动到 low_quality 文件夹，并记录原因。"""
    low_quality_dir = os.path.join(root_dir, "low_quality")
    if not os.path.exists(low_quality_dir):
        os.makedirs(low_quality_dir)
        
    dest_path = os.path.join(low_quality_dir, os.path.basename(file_path))
    
    # 处理重名
    if os.path.exists(dest_path):
        base, ext = os.path.splitext(os.path.basename(file_path))
        counter = 1
        while os.path.exists(os.path.join(low_quality_dir, f"{base}_{counter}{ext}")):
            counter += 1
        dest_path = os.path.join(low_quality_dir, f"{base}_{counter}{ext}")
        
    try:
        shutil.move(file_path, dest_path)
        return True
    except Exception as e:
        print(f"移动失败 {file_path}: {e}")
        return False

def process_file(file_path, args, root_dir):
    """处理单个文件：检查并移动。返回 (文件名, 原因) 或 (None, None)。"""
    passed, reason = check_image_quality(file_path, args)
    if not passed:
        if move_to_low_quality(file_path, root_dir, reason):
            return os.path.basename(file_path), reason
    return None, None

def main():
    parser = argparse.ArgumentParser(description="筛选低质量图片（分辨率、大小、模糊度、噪声、对比度）。", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dir", type=str, default=".", help="要扫描的目录")
    parser.add_argument("--threads", type=int, default=4, help="线程数 (默认: 4)")
    parser.add_argument("--min-width", type=int, default=512, help="最小宽度 (默认: 512)")
    parser.add_argument("--min-height", type=int, default=512, help="最小高度 (默认: 512)")
    parser.add_argument("--min-size", type=float, default=50, help="最小文件大小 KB (默认: 50)")
    
    parser.add_argument("--blur-threshold", type=float, default=0, help="模糊度阈值 (默认: 0, 不检测)。\n建议值: 100-300。低于此值视为模糊。")
    parser.add_argument("--max-noise", type=float, default=0, help="最大噪声阈值 (默认: 0, 不检测)。\n建议值: 20-50。高于此值视为噪点多。")
    parser.add_argument("--min-contrast", type=float, default=0, help="最小对比度/像素标准差 (默认: 0, 不检测)。\n建议值: 10-20。低于此值视为灰蒙蒙/纯色。")
    parser.add_argument("--max-contrast", type=float, default=0, help="最大对比度/像素标准差 (默认: 0, 不检测)。\n高于此值视为异常。")
    
    args = parser.parse_args()
    
    directory = os.path.abspath(args.dir)
    print(f"扫描目录: {directory}")
    print(f"线程数: {args.threads}")
    print("正在筛选...")
    
    all_files = []
    for root, _, files in os.walk(directory):
        if "low_quality" in root or "duplicates" in root:
            continue
        for file in files:
            if is_image_file(file):
                all_files.append(os.path.join(root, file))
                
    print(f"找到 {len(all_files)} 张图片，开始检测...")
    
    moved_count = 0
    reasons_count = {}
    
    # 使用多线程处理
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # 包装函数以传递固定参数
        func = lambda f: process_file(f, args, directory)
        
        # 使用 tqdm 显示进度，executor.map 按顺序返回结果
        for fname, reason in tqdm(executor.map(func, all_files), total=len(all_files), unit="img"):
            if fname:
                moved_count += 1
                print(f"  [移出] {fname}: {reason}")
                
                base_reason = reason.split('(')[0].strip()
                reasons_count[base_reason] = reasons_count.get(base_reason, 0) + 1

    print("\n筛选完成！")
    print(f"共扫描: {len(all_files)}")
    print(f"共移出: {moved_count}")
    print("移出原因统计:")
    for r, c in reasons_count.items():
        print(f"  - {r}: {c} 张")
    print(f"低质量图片已移动到: {os.path.join(directory, 'low_quality')}")

if __name__ == "__main__":
    main()
