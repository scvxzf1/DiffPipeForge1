"""
图像宽高比统计脚本
统计数据集中图像的宽高比分布
"""


"""
基本用法 (使用预设分桶)
python aspect_ratio_stats.py 

使用自定义分桶，桶大小为0.2
python aspect_ratio_stats.py <图像目录路径> --mode custom --bucket-size 0.2

指定输出文件
python aspect_ratio_stats.py <图像目录路径> --output my_stats.json
"""


import os
import argparse
from pathlib import Path
from PIL import Image
from collections import defaultdict
import json
import cv2  # Added for video support


def get_aspect_ratio(width, height):
    """计算宽高比"""
    if height == 0: return 0
    return width / height


def classify_aspect_ratio(ratio):
    """
    将宽高比分类到预定义的桶中
    """
    if ratio > 2.0:
        return "超宽 (>2.0)"
    elif ratio >= 1.8:
        return "宽屏 (1.8-2.0)"
    elif ratio >= 1.7:
        return "16:9 (1.7-1.8)"
    elif ratio >= 1.4:
        return "3:2 (1.4-1.7)"
    elif ratio >= 1.1:
        return "4:3 (1.1-1.4)"
    elif ratio >= 0.9:
        return "方形 (0.9-1.1)"
    elif ratio >= 0.7:
        return "3:4 (0.7-0.9)"
    elif ratio >= 0.5:
        return "竖屏 (0.5-0.7)"
    else:
        return "超竖 (<0.5)"


def custom_buckets(ratio, bucket_size=0.1):
    """
    使用自定义桶大小进行分类
    """
    bucket = round(ratio / bucket_size) * bucket_size
    return f"{bucket:.2f}"


def analyze_files(target_dir, file_type='image', bucket_mode='preset', bucket_size=0.1):
    """
    分析指定目录下的所有文件（图像或视频）
    
    Args:
        target_dir: 目录路径
        file_type: 'image' 或 'video'
        bucket_mode: 分桶模式 ('preset' 或 'custom')
        bucket_size: 自定义桶大小
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.m4v'}
    
    target_extensions = image_extensions if file_type == 'image' else video_extensions
    
    stats = defaultdict(int)
    ratio_list = []
    total_files = 0
    error_count = 0
    
    print(f"正在扫描目录 ({file_type}): {target_dir}")
    print("-" * 60)
    
    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(target_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()
            
            if ext in target_extensions:
                width, height = 0, 0
                try:
                    if file_type == 'image':
                        with Image.open(file_path) as img:
                            width, height = img.size
                    else: # video
                        cap = cv2.VideoCapture(file_path)
                        if cap.isOpened():
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            cap.release()
                        else:
                            raise Exception("Could not open video file")

                    if width > 0 and height > 0:
                        ratio = get_aspect_ratio(width, height)
                        ratio_list.append(ratio)
                        
                        # 根据模式分类
                        if bucket_mode == 'preset':
                            bucket = classify_aspect_ratio(ratio)
                        else:
                            bucket = custom_buckets(ratio, bucket_size)
                        
                        stats[bucket] += 1
                        total_files += 1
                        
                        if total_files % 100 == 0:
                            print(f"已处理 {total_files} 个文件...")
                    
                except Exception as e:
                    error_count += 1
                    # print(f"处理文件时出错 {filename}: {str(e)}") # Reduce noise
    
    return stats, ratio_list, total_files, error_count


def print_statistics(stats, ratio_list, total_files, error_count):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print(f"统计完成！")
    print(f"总文件数: {total_files}")
    print(f"错误数: {error_count}")
    print("=" * 60)
    
    if not ratio_list:
        print("没有找到有效的文件")
        return
    
    # 计算统计数据
    avg_ratio = sum(ratio_list) / len(ratio_list)
    min_ratio = min(ratio_list)
    max_ratio = max(ratio_list)
    
    print(f"\n宽高比统计:")
    print(f"  平均值: {avg_ratio:.3f}")
    print(f"  最小值: {min_ratio:.3f}")
    print(f"  最大值: {max_ratio:.3f}")
    
    # 打印分布情况
    print(f"\n宽高比分布:")
    print("-" * 60)
    print(f"{'分桶':<20} {'数量':<10} {'百分比':<10} {'可视化'}")
    print("-" * 60)
    
    # 排序桶 logic (same as before)
    bucket_order = {
        "超宽 (>2.0)": 2.5,
        "宽屏 (1.8-2.0)": 1.9,
        "16:9 (1.7-1.8)": 1.75,
        "3:2 (1.4-1.7)": 1.55,
        "4:3 (1.1-1.4)": 1.25,
        "方形 (0.9-1.1)": 1.0,
        "3:4 (0.7-0.9)": 0.8,
        "竖屏 (0.5-0.7)": 0.6,
        "超竖 (<0.5)": 0.25
    }
    
    def get_sort_key(item):
        bucket_name = item[0]
        if bucket_name in bucket_order:
            return bucket_order[bucket_name]
        try:
            return float(bucket_name)
        except ValueError:
            return 0
    
    sorted_stats = sorted(stats.items(), key=get_sort_key, reverse=True)
    
    for bucket, count in sorted_stats:
        percentage = (count / total_files) * 100
        bar_length = int(percentage / 2)
        bar = "█" * bar_length
        print(f"{bucket:<20} {count:<10} {percentage:>6.2f}%   {bar}")
    
    print("-" * 60)


def sorted_stats_for_json(stats, ratio_list):
    if not ratio_list:
        return []
    
    bucket_order = {
        "超宽 (>2.0)": 2.5,
        "宽屏 (1.8-2.0)": 1.9,
        "16:9 (1.7-1.8)": 1.75,
        "3:2 (1.4-1.7)": 1.55,
        "4:3 (1.1-1.4)": 1.25,
        "方形 (0.9-1.1)": 1.0,
        "3:4 (0.7-0.9)": 0.8,
        "竖屏 (0.5-0.7)": 0.6,
        "超竖 (<0.5)": 0.25
    }
    
    def get_sort_key(item):
        bucket_name = item[0]
        if bucket_name in bucket_order:
            return bucket_order[bucket_name]
        try:
            return float(bucket_name)
        except ValueError:
            return 0
    
    sorted_items = sorted(stats.items(), key=get_sort_key, reverse=True)
    
    res = []
    total = len(ratio_list)
    for bucket, count in sorted_items:
        res.append({
            "bucket": bucket,
            "count": count,
            "percentage": (count / total) * 100
        })
    return res


def main():
    parser = argparse.ArgumentParser(description='统计数据集的宽高比分布')
    parser.add_argument('--dir', type=str, required=True, help='目录路径')
    parser.add_argument('--type', type=str, choices=['image', 'video'], default='image', help='文件类型: image 或 video')
    parser.add_argument('--mode', type=str, choices=['preset', 'custom'], default='preset', help='分桶模式')
    parser.add_argument('--bucket-size', type=float, default=0.1, help='自定义桶大小')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"错误: 目录不存在: {args.dir}")
        return
    
    stats, ratio_list, total_files, error_count = analyze_files(
        args.dir,
        file_type=args.type,
        bucket_mode=args.mode, 
        bucket_size=args.bucket_size
    )
    
    print_statistics(stats, ratio_list, total_files, error_count)
    
    results = {
        "total": total_files,
        "errors": error_count,
        "avg": sum(ratio_list) / len(ratio_list) if ratio_list else 0,
        "min": min(ratio_list) if ratio_list else 0,
        "max": max(ratio_list) if ratio_list else 0,
        "distribution": sorted_stats_for_json(stats, ratio_list)
    }
    print(f"\n__ASPECT_STATS_JSON_START__\n{json.dumps(results, ensure_ascii=False)}\n__ASPECT_STATS_JSON_END__")


if __name__ == '__main__':
    main()
