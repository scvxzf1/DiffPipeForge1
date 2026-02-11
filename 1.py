import torch

# 1. 检查PyTorch的CUDA支持状态
print("PyTorch CUDA是否可用:", torch.cuda.is_available())
# 2. 查看CUDA版本（若可用）
if torch.cuda.is_available():
    print("CUDA版本:", torch.version.cuda)
    print("可用GPU数量:", torch.cuda.device_count())
    print("当前GPU名称:", torch.cuda.get_device_name(0))
else:
    print("⚠️ PyTorch未检测到CUDA，即使系统装了CUDA也没用")