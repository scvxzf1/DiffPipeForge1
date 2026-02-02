import os
import hashlib
import shutil
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
import imagehash


# 找相似图：python deduplicate_images.py --mode similar
# 找完全重复：python deduplicate_images.py --mode exact 
# 您可以通过 --threshold 参数调整灵敏度（默认 5，越小越严格）。 --threshold 越大，相似度越低

def calculate_md5(file_path, block_size=65536):
    """计算文件的 MD5 哈希值。"""
    md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                md5.update(block)
        return md5.hexdigest()
    except OSError:
        return None

def calculate_phash(file_path):
    """计算图片的感知哈希 (pHash)。"""
    try:
        with Image.open(file_path) as img:
            return imagehash.phash(img)
    except Exception:
        return None

def find_exact_duplicates(all_files, duplicates_folder):
    """基于 MD5 查找并移动完全重复的图片。"""
    print("正在计算 MD5 哈希...")
    files_by_hash = defaultdict(list)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda f: (f, calculate_md5(f)), all_files), total=len(all_files), unit="img"))

    for file_path, file_hash in results:
        if file_hash:
            files_by_hash[file_hash].append(file_path)

    process_duplicates(files_by_hash, duplicates_folder, "完全重复 (Exact)")

def find_similar_images(all_files, duplicates_folder, threshold=5):
    """基于感知哈希 (pHash) 查找并移动相似图片。"""
    print("正在计算感知哈希 (pHash)...")
    # pHash 计算较慢，使用并行处理
    hashes = []
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda f: (f, calculate_phash(f)), all_files), total=len(all_files), unit="img"))
    
    # 过滤掉计算失败的哈希
    valid_results = [r for r in results if r[1] is not None]
    
    print(f"正在比对哈希 (阈值: {threshold})...")
    
    # 简单的贪心算法：
    # 1. 对文件进行排序（以保持一致的“原图”判定）
    # 2. 迭代并维护一个“唯一”图片列表。
    # 3. 对于每张新图片，与“唯一”列表进行比对。如果足够相似，则视为重复。
    
    valid_results.sort(key=lambda x: (len(os.path.basename(x[0])), os.path.basename(x[0])))
    
    unique_images = [] # 存储 (路径, 哈希)
    moved_count = 0
    
    for file_path, file_hash in tqdm(valid_results, unit="img"):
        is_duplicate = False
        for unique_path, unique_hash in unique_images:
            if file_hash - unique_hash <= threshold:
                # 发现相似图片
                is_duplicate = True
                try:
                    move_duplicate(file_path, duplicates_folder)
                    print(f"  与 {os.path.basename(unique_path)} 相似: {os.path.basename(file_path)} (距离: {file_hash - unique_hash})")
                    moved_count += 1
                except Exception as e:
                    print(f"  移动 {os.path.basename(file_path)} 时出错: {e}")
                break
        
        if not is_duplicate:
            unique_images.append((file_path, file_hash))
            
    print(f"共移动了 {moved_count} 张相似图片。")

def process_duplicates(files_by_hash, duplicates_folder, mode_name):
    """根据哈希分组移动重复文件。"""
    moved_count = 0
    duplicate_groups = 0
    
    print(f"\n正在处理 {mode_name} 重复项...")
    
    for file_hash, file_list in files_by_hash.items():
        if len(file_list) > 1:
            duplicate_groups += 1
            # 排序：文件名最短的排前面（假设为原图）
            file_list.sort(key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))
            
            original = file_list[0]
            duplicates = file_list[1:]
            
            print(f"保留原图: {os.path.basename(original)}")
            
            for dup in duplicates:
                try:
                    move_duplicate(dup, duplicates_folder)
                    print(f"  移动重复项: {os.path.basename(dup)}")
                    moved_count += 1
                except Exception as e:
                    print(f"  移动 {os.path.basename(dup)} 时出错: {e}")

    print(f"共移动了 {moved_count} 张 {mode_name} 重复图片。")

def move_duplicate(file_path, duplicates_folder):
    """将单个文件移动到 duplicates 文件夹，处理文件名冲突。"""
    dest_path = os.path.join(duplicates_folder, os.path.basename(file_path))
    if os.path.exists(dest_path):
        base, ext = os.path.splitext(os.path.basename(file_path))
        counter = 1
        while os.path.exists(os.path.join(duplicates_folder, f"{base}_{counter}{ext}")):
            counter += 1
        dest_path = os.path.join(duplicates_folder, f"{base}_{counter}{ext}")
    shutil.move(file_path, dest_path)

def main():
    parser = argparse.ArgumentParser(description="使用 MD5 (完全相同) 或 pHash (相似) 对图片进行去重。")
    parser.add_argument("--dir", type=str, default=".", help="要扫描的目录")
    parser.add_argument("--mode", type=str, choices=["exact", "similar"], default="exact", help="去重模式")
    parser.add_argument("--threshold", type=int, default=5, help="相似度判定的汉明距离阈值 (默认: 5)")
    
    args = parser.parse_args()
    
    directory = os.path.abspath(args.dir)
    duplicates_folder = os.path.join(directory, "duplicates")
    
    if not os.path.exists(duplicates_folder):
        os.makedirs(duplicates_folder)
        
    print(f"扫描目录: {directory}")
    print(f"模式: {args.mode}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    all_files = []
    
    for root, _, files in os.walk(directory):
        # 跳过 duplicates 文件夹本身
        if "duplicates" in root:
            continue
            
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                all_files.append(os.path.join(root, file))

    print(f"找到 {len(all_files)} 张图片。")
    
    if args.mode == "exact":
        find_exact_duplicates(all_files, duplicates_folder)
    elif args.mode == "similar":
        find_similar_images(all_files, duplicates_folder, args.threshold)

if __name__ == "__main__":
    main()
