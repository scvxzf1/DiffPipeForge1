import os
import shutil
from pathlib import Path

def find_and_move_untagged_images(source_dir='.', target_subdir='unprompt'):
    """
    遍历目录中的所有图片，找出没有对应txt文件的图片，移动到子文件夹
    
    Args:
        source_dir: 源目录路径，默认为当前目录
        target_subdir: 目标子文件夹名称，默认为'unprompt'
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff'}
    
    # 转换为Path对象
    source_path = Path(source_dir).resolve()
    target_path = source_path / target_subdir
    
    # 创建目标文件夹（如果不存在）
    target_path.mkdir(exist_ok=True)
    
    # 统计信息
    total_images = 0
    untagged_images = 0
    moved_images = 0
    
    print(f"开始扫描目录: {source_path}")
    print(f"目标文件夹: {target_path}")
    print("-" * 60)
    
    # 遍历源目录中的所有文件
    for file_path in source_path.iterdir():
        # 跳过目录
        if file_path.is_dir():
            continue
        
        # 检查是否是图片文件
        if file_path.suffix.lower() in image_extensions:
            total_images += 1
            
            # 检查是否存在对应的txt文件
            txt_path = file_path.with_suffix('.txt')
            
            if not txt_path.exists():
                untagged_images += 1
                print(f"发现未打标图片: {file_path.name}")
                
                # 移动文件到目标文件夹
                try:
                    target_file = target_path / file_path.name
                    
                    # 如果目标文件已存在，添加序号
                    if target_file.exists():
                        counter = 1
                        while target_file.exists():
                            new_name = f"{file_path.stem}_{counter}{file_path.suffix}"
                            target_file = target_path / new_name
                            counter += 1
                    
                    shutil.move(str(file_path), str(target_file))
                    moved_images += 1
                    print(f"  -> 已移动到: {target_file.relative_to(source_path)}")
                    
                except Exception as e:
                    print(f"  -> 移动失败: {e}")
    
    print("-" * 60)
    print(f"扫描完成!")
    print(f"总图片数: {total_images}")
    print(f"未打标图片数: {untagged_images}")
    print(f"成功移动: {moved_images}")
    
    if moved_images > 0:
        print(f"\n所有未打标图片已移动到: {target_path}")

if __name__ == '__main__':
    # 使用当前目录，也可以修改为指定目录
    # 例如: find_and_move_untagged_images('C:/Users/LUYUE/Desktop/1')
    find_and_move_untagged_images()
