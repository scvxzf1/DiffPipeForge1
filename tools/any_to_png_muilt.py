import os
import threading
import argparse
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
import multiprocessing

"""任何图片转为png，多线程处理"""

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageConverter:
    def __init__(self, target_format='png', delete_source=True):
        self.converted_count = 0
        self.error_count = 0
        self.converted_lock = threading.Lock()
        self.error_lock = threading.Lock()
        self.processed_files = set()
        self.processed_lock = threading.Lock()
        self.delete_source = delete_source
        self.target_format = target_format.lower().replace('jpeg', 'jpg')
        if self.target_format == 'jpg':
             self.target_ext = '.jpg'
        else:
             self.target_ext = f'.{self.target_format}'
        
        # 支持的输入格式
        self.supported_input_formats = {'.webp', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.png'}
        
    def process_single_file(self, file_path):
        """处理单个文件的转换逻辑"""
        file_ext = file_path.suffix.lower()
        
        # 检查是否支持该格式，且不是目标格式本身
        if file_ext not in self.supported_input_formats or (file_ext == self.target_ext or (self.target_format == 'jpg' and file_ext == '.jpeg')):
            return
        
        # 检查文件是否正在被其他线程处理
        with self.processed_lock:
            if file_path in self.processed_files:
                return
            self.processed_files.add(file_path)
        
        try:
            with Image.open(file_path) as img:
                # 构建新的文件名
                new_file_path = file_path.with_suffix(self.target_ext)
                
                # 如果目标文件已存在，跳过
                if new_file_path.exists():
                    logger.warning(f"目标文件已存在，跳过: {new_file_path}")
                    return
                
                # 转换逻辑
                save_format = self.target_format.upper()
                if save_format == 'JPG':
                    save_format = 'JPEG'
                
                save_kwargs = {'optimize': True}
                
                # 处理不同格式的需求
                if self.target_format == 'jpg':
                    # JPG 不支持透明，必须转为 RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    save_kwargs['quality'] = 95
                elif self.target_format == 'webp':
                    save_kwargs['quality'] = 90
                elif self.target_format == 'png':
                    if img.mode in ('RGBA', 'LA', 'P'):
                        if img.mode == 'P' and 'transparency' in img.info:
                            img = img.convert('RGBA')
                    else:
                        img = img.convert('RGB')
                
                # 保存
                img.save(new_file_path, save_format, **save_kwargs)
                
                # 验证新文件是否创建成功
                if new_file_path.exists() and new_file_path.stat().st_size > 0:
                    # 根据配置决定是否删除原文件
                    if self.delete_source:
                        file_path.unlink()
                    
                    with self.converted_lock:
                        self.converted_count += 1
                        print(f"Success: {file_path.name} -> {new_file_path.name}") # Stdout for UI
                else:
                    raise Exception("新文件创建失败或为空")
                    
        except Exception as e:
            with self.error_lock:
                self.error_count += 1
            print(f"Error {file_path.name}: {str(e)}")
            
            # 清理可能创建的不完整文件
            target_path = file_path.with_suffix(self.target_ext)
            if target_path.exists():
                try:
                    target_path.unlink()
                except:
                    pass

def convert_images(directory, target_format='png', max_workers=None, delete_source=True):
    """
    将指定目录下的所有图片转换为目标格式（多线程版本）
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.error(f"目录不存在: {directory}")
        return
    
    if not directory_path.is_dir():
        logger.error(f"路径不是目录: {directory}")
        return
    
    # Init converter
    converter = ImageConverter(target_format=target_format, delete_source=delete_source)
    
    # 获取所有需要处理的文件
    files_to_process = []
    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in converter.supported_input_formats:
            # Check if it's already the target format
            if file_path.suffix.lower() == converter.target_ext:
                continue
            if converter.target_format == 'jpg' and file_path.suffix.lower() == '.jpeg':
                continue
            files_to_process.append(file_path)
    
    if not files_to_process:
        logger.info("没有找到可转换的图片文件")
        print("No convertible images found.")
        return
    
    logger.info(f"找到 {len(files_to_process)} 个待转换文件，目标格式: {target_format}")
    print(f"Found {len(files_to_process)} files to convert to {target_format.upper()}.")
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        list(executor.map(converter.process_single_file, files_to_process))
    
    # 输出统计信息
    print(f"\nconversion_finished: {converter.converted_count} success, {converter.error_count} failed")
    logger.info(f"\n转换完成!")
    logger.info(f"成功转换: {converter.converted_count} 个文件")
    if converter.error_count > 0:
        logger.warning(f"失败: {converter.error_count} 个文件")

def main():
    parser = argparse.ArgumentParser(description='Convert images to target format')
    parser.add_argument('--dir', type=str, help='Directory containing images')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'jpg', 'webp'], help='Target format (png, jpg, webp)')
    parser.add_argument('--threads', type=int, default=None, help='Number of threads')
    parser.add_argument('--delete', action='store_true', help='Delete source files after conversion')
    
    args = parser.parse_args()
    
    # Interactive mode if no dir provided
    if not args.dir:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        print(f"图片格式转换工具 (Interactive Mode)")
        print("=" * 50)
        print(f"工作目录: {current_directory}")
        print(f"目标格式: {args.format.upper()}")
        print("=" * 50)
        
        confirm = input(f"确认开始转换当前目录为 {args.format.upper()}? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("操作已取消")
            return
            
        max_workers = min(multiprocessing.cpu_count() * 2, 20)
        convert_images(current_directory, target_format=args.format, max_workers=max_workers, delete_source=True)
    else:
        # CLI Mode
        workers = args.threads if args.threads else min(multiprocessing.cpu_count() * 2, 20)
        convert_images(args.dir, target_format=args.format, max_workers=workers, delete_source=args.delete)

if __name__ == "__main__":
    main()
