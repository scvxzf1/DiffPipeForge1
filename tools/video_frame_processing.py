import cv2
import os
import re
import subprocess
import argparse

# 默认设置
DEFAULT_TARGET_FPS = 15
OUTPUT_DIR = 'low_fps_videos'

def natural_keys(text):
    '''
    自然排序算法: 1.mp4, 2.mp4, 10.mp4...
    '''
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def get_video_files(directory):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    video_files = [f for f in files if os.path.splitext(f)[1].lower() in video_extensions]
    video_files.sort(key=natural_keys)
    return video_files

def scan_videos(directory, files):
    """
    统计并显示视频信息，包括分辨率、总帧数、总时长和帧数分布
    """
    total_frames_all_videos = 0
    total_duration_seconds = 0
    frame_counts = []
    
    print("\n" + "=" * 95)
    print(f"{'文件名':<30} | {'分辨率':<15} | {'帧数':<10} | {'帧率':<10} | {'时长(秒)':<10}")
    print("-" * 95)
    
    for file in files:
        file_path = os.path.join(directory, file)
        try:
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                print(f"{file:<30} | {'-':<15} | {'错误':<10} | {'-':<10} | {'-':<10}")
                continue
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution = f"{width}x{height}"
            
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frames / fps if fps > 0 else 0
            
            print(f"{file:<30} | {resolution:<15} | {frames:<10} | {fps:<10.2f} | {duration:<10.2f}")
            
            total_frames_all_videos += frames
            total_duration_seconds += duration
            frame_counts.append(frames)
                
            cap.release()
        except Exception as e:
            print(f"读取 {file} 时出错: {e}")
            
    print("-" * 95)
    
    # 打印总结信息
    total_videos = len(files)
    hours = int(total_duration_seconds // 3600)
    minutes = int((total_duration_seconds % 3600) // 60)
    seconds = int(total_duration_seconds % 60)
    
    print("【统计总结】")
    print(f"视频总数: {total_videos}")
    print(f"总帧数:   {total_frames_all_videos}")
    print(f"总时长:   {total_duration_seconds:.2f} 秒 ({hours}小时 {minutes}分 {seconds}秒)")
    
    if frame_counts:
        print("-" * 40)
        print("【帧数分布】")
        print(f"{'区间范围':<20} | {'数量':<10}")
        print("-" * 35)

        max_frames = max(frame_counts)
        # 动态决定区间大小
        if max_frames <= 100:
            step = 10
        elif max_frames <= 1000:
            step = 100
        elif max_frames <= 10000:
            step = 1000
        else:
            step = 5000

        distribution = {}
        for f in frame_counts:
            lower_bound = (f // step) * step
            upper_bound = lower_bound + step
            key = (lower_bound, upper_bound)
            distribution[key] = distribution.get(key, 0) + 1
        
        for (start, end) in sorted(distribution.keys()):
            range_str = f"[{start}, {end})"
            print(f"{range_str:<20} | {distribution[(start, end)]:<10}")
        
    print("=" * 95 + "\n")
    
    # 输出 JSON 格式统计信息供前端 UI 使用
    import json
    stats_data = {
        "total_videos": total_videos,
        "total_frames": total_frames_all_videos,
        "total_duration": total_duration_seconds,
        "resolution_distribution": {}, # 可以扩展
        "frame_distribution": []
    }
    
    if frame_counts:
        for (start, end), count in sorted(distribution.items()):
            stats_data["frame_distribution"].append({
                "range": f"[{start}, {end})",
                "count": count,
                "percentage": (count / total_videos) * 100
            })
            
    print("__VIDEO_STATS_JSON_START__")
    print(json.dumps(stats_data, ensure_ascii=False, indent=2))
    print("__VIDEO_STATS_JSON_END__")
    
    return total_frames_all_videos

def convert_videos(directory, files, target_fps):
    """
    降低视频帧率
    """
    output_path = os.path.join(directory, OUTPUT_DIR)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"创建输出目录: {output_path}")
    else:
        print(f"输出目录已存在: {output_path}")

    print(f"即将开始转换，目标帧率: {target_fps} FPS")
    print("-" * 80)
    
    success_count = 0
    total_count = len(files)

    for i, file in enumerate(files, 1):
        input_file = os.path.join(directory, file)
        output_file = os.path.join(output_path, file)
        
        cmd = [
            'ffmpeg', 
            '-i', input_file, 
            '-r', str(target_fps),
            '-c:v', 'libx264',   
            '-pix_fmt', 'yuv420p', 
            '-c:a', 'copy',      
            '-y',                
            output_file
        ]
        
        try:
            print(f"[{i}/{total_count}] 正在处理 {file} ...", end='', flush=True)
            # 运行命令，隐藏详细输出
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"\r[{i}/{total_count}] {file:<30} | 完成     ")
            success_count += 1
        except subprocess.CalledProcessError:
            print(f"\r[{i}/{total_count}] {file:<30} | 失败     ")
        except FileNotFoundError:
            print(f"\n错误: 未找到 ffmpeg。请确保系统已安装 ffmpeg 并添加到环境变量。")
            return

    print("-" * 80)
    print(f"任务完成。成功转换: {success_count}/{total_count}")
    print(f"文件保存在: {os.path.abspath(output_path)}")

def main():
    parser = argparse.ArgumentParser(description="视频管理工具: 查看帧数或降低帧率")
    parser.add_argument('--reduce', action='store_true', help="启用降低帧率模式")
    parser.add_argument('--fps', type=int, default=DEFAULT_TARGET_FPS, help=f"设置目标帧率 (默认: {DEFAULT_TARGET_FPS})")
    parser.add_argument('--path', type=str, default='.', help="设置视频文件夹路径 (默认: 当前目录)")
    
    args = parser.parse_args()
    
    #确定工作目录
    working_directory = args.path
    if working_directory == '.':
        working_directory = os.getcwd()
    
    # 检查路径是否存在
    if not os.path.isdir(working_directory):
        print(f"错误: 路径 '{working_directory}' 不存在或不是一个目录。")
        return

    video_files = get_video_files(working_directory)
    
    if not video_files:
        print(f"在目录 '{working_directory}' 中未找到视频文件。")
        return

    if args.reduce:
        # 降低帧率模式
        convert_videos(working_directory, video_files, args.fps)
    else:
        # 默认模式：查看信息
        print(f"当前工作目录: {working_directory}")
        scan_videos(working_directory, video_files)

if __name__ == "__main__":
    main()
