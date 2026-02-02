import os
import argparse
from pathlib import Path

def rename_files_sequentially(directory, start_num=1, prefix="", extension=None):
    """
    按顺序重命名目录下的所有文件
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        print(f"错误: 目录不存在 - {directory}")
        return

    # 获取所有文件
    files = []
    for file_path in directory_path.iterdir():
        if file_path.is_file():
            # 跳过脚本自身
            if file_path.name == os.path.basename(__file__):
                continue
            
            # 如果指定了扩展名，只处理该类型文件
            if extension is None or extension == "all" or file_path.suffix.lower() == extension.lower():
                files.append(file_path)
    
    # 按文件名排序
    files.sort(key=lambda x: x.name)
    
    if not files:
        print("没有找到需要重命名的文件")
        return
    
    print(f"找到 {len(files)} 个文件需要重命名")
    print("=" * 60)
    
    # 先重命名为临时名称，避免文件名冲突
    temp_files = []
    for i, file_path in enumerate(files):
        temp_name = f"__temp_{i}_{file_path.suffix}"
        temp_path = file_path.parent / temp_name
        try:
            file_path.rename(temp_path)
            temp_files.append((temp_path, file_path.suffix))
        except Exception as e:
            print(f"临时重命名失败: {file_path.name} -> {e}")
    
    # 再重命名为最终名称
    renamed_count = 0
    for i, (temp_path, suffix) in enumerate(temp_files, start=start_num):
        try:
            new_name = f"{prefix}{i}{suffix}"
            new_path = temp_path.parent / new_name
            temp_path.rename(new_path)
            renamed_count += 1
            print(f"{i-start_num+1}. 重命名为: {new_name}")
        except Exception as e:
            print(f"重命名失败: {temp_path.name} -> {e}")
    
    print("=" * 60)
    print(f"重命名完成! 共处理 {renamed_count} 个文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="文件顺序重命名工具")
    parser.add_argument("--dir", type=str, required=True, help="目标目录")
    parser.add_argument("--prefix", type=str, default="", help="文件名前缀")
    parser.add_argument("--start", type=int, default=1, help="起始编号")
    parser.add_argument("--ext", type=str, default="all", help="只重命名指定扩展名的文件 (例如 .png, .webp)")

    args = parser.parse_args()
    
    rename_files_sequentially(
        directory=args.dir,
        start_num=args.start,
        prefix=args.prefix,
        extension=args.ext if args.ext != "all" else None
    )
