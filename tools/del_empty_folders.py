import os
import sys

def remove_empty_folders(path=None, remove_current=True):
    """
    删除指定路径下的所有空文件夹
    
    Args:
        path: 要清理的文件夹路径，默认为当前目录
        remove_current: 是否删除当前目录（如果为空）
    """
    if path is None:
        path = os.getcwd()
    
    print(f"正在扫描目录: {path}")
    
    # 计数器
    removed_count = 0
    error_count = 0
    
    try:
        # 使用os.walk遍历所有子目录（自底向上）
        for root, dirs, files in os.walk(path, topdown=False):
            # 跳过隐藏文件夹（可选）
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            # 检查当前目录是否为空
            if not dirs and not files:
                try:
                    os.rmdir(root)
                    print(f"✓ 删除空文件夹: {root}")
                    removed_count += 1
                except OSError as e:
                    print(f"✗ 删除失败 {root}: {e}")
                    error_count += 1
                except Exception as e:
                    print(f"✗ 未知错误 {root}: {e}")
                    error_count += 1
        
        # 检查并处理当前目录（如果允许）
        if remove_current and path == os.getcwd():
            if not os.listdir(path):
                try:
                    os.rmdir(path)
                    print(f"✓ 删除当前空文件夹: {path}")
                    removed_count += 1
                except OSError as e:
                    print(f"✗ 无法删除当前目录 {path}: {e}")
                    error_count += 1
        
        # 输出统计信息
        print("\n" + "="*50)
        print(f"清理完成!")
        print(f"删除的空文件夹数量: {removed_count}")
        print(f"删除失败的数量: {error_count}")
        
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n发生错误: {e}")
        sys.exit(1)

def main():
    """主函数"""
    print("空文件夹清理工具")
    print("=" * 50)
    
    # 获取用户输入
    path = input("请输入要清理的目录路径 (直接回车使用当前目录): ").strip()
    if not path:
        path = os.getcwd()
    
    # 验证路径是否存在
    if not os.path.exists(path):
        print(f"错误: 路径不存在 - {path}")
        sys.exit(1)
    
    if not os.path.isdir(path):
        print(f"错误: 路径不是目录 - {path}")
        sys.exit(1)
    
    # 确认操作
    print(f"\n将清理目录: {path}")
    confirm = input("确认开始清理? (y/N): ").strip().lower()
    
    if confirm not in ['y', 'yes']:
        print("操作已取消")
        sys.exit(0)
    
    # 执行清理
    remove_empty_folders(path)

if __name__ == "__main__":
    main()