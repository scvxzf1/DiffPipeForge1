import argparse
import os
import json
import torch
import torch.nn as nn
from collections import defaultdict
from pathlib import Path
from safetensors.torch import load_file
import glob
import gc

# 尝试导入 rich
try:
    from rich.console import Console
    from rich.table import Table
    from rich.tree import Tree
    from rich import print as rprint
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    print("提示: 建议安装 'rich' 库以获得最佳显示效果: pip install rich")

def format_size(num_params):
    if num_params >= 1e9: return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6: return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3: return f"{num_params / 1e3:.2f}K"
    return str(num_params)

class ModelAnalyzer:
    def __init__(self, show_all_keys=False):
        self.show_all_keys = show_all_keys
        # 全局统计容器
        self.global_tree = defaultdict(lambda: {"params": 0})
        self.global_total_params = 0
        self.global_keys = []
        self.scanned_files = []

    def analyze_diffusers(self, model_path):
        print(f"\n======== 分析 Diffusers 模型: {model_path} ========\n")
        
        # 1. 打印 Config (如果有)
        self.print_config(model_path)
        
        # 2. 尝试标准加载 (为了兼容非自定义模型)
        try:
            from diffusers import UNet2DConditionModel, Transformer2DModel
            import logging
            # 抑制 diffusers 的报错日志
            logging.getLogger("diffusers").setLevel(logging.ERROR)
            
            # 尝试 Transformer
            if os.path.exists(os.path.join(model_path, "transformer")):
                try:
                    # 仅作尝试，不强求成功
                    model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer")
                    print("✅ 成功通过 Diffusers 加载架构 (Transformer2DModel)")
                    self.analyze_live_model(model)
                    return
                except: pass
        except ImportError:
            pass

        print("⚠️ 标准加载失败或跳过 (检测为自定义架构/分卷权重)。")
        print("🔄 切换到 [分卷合并扫描模式] ...")
        self.scan_sharded_weights(model_path)

    def scan_sharded_weights(self, model_path):
        # 搜索逻辑：优先找 transformer 文件夹，其次 unet，最后根目录
        search_paths = [
            os.path.join(model_path, "transformer"),
            os.path.join(model_path, "unet"),
            model_path
        ]
        
        target_files = []
        
        # 1. 定位包含权重的文件夹
        for p in search_paths:
            if not os.path.exists(p): continue
            
            # 查找 safetensors
            files = glob.glob(os.path.join(p, "*.safetensors"))
            # 过滤掉 optimizer 或 text_encoder (通常我们只关心主模型)
            files = [f for f in files if "optimizer" not in f and "text_encoder" not in f]
            
            if files:
                target_files = files
                print(f"📂 在文件夹发现权重: {p}")
                break
        
        if not target_files:
            print("❌ 未找到任何 .safetensors 权重文件。")
            return

        target_files.sort()
        print(f"📦 检测到 {len(target_files)} 个权重分片，开始逐个分析...")
        
        # 2. 逐个文件读取并合并信息
        for i, file_path in enumerate(target_files):
            file_name = os.path.basename(file_path)
            print(f"   [{i+1}/{len(target_files)}] 读取: {file_name} ...", end="\r")
            try:
                self.process_single_file_content(file_path)
            except Exception as e:
                print(f"\n   ❌ 读取失败 {file_name}: {e}")
            
            # 强制垃圾回收，防止内存爆掉
            gc.collect()

        print(f"\n✅ 扫描完成！共分析 {len(self.scanned_files)} 个文件。\n")
        self.print_final_report()

    def process_single_file_content(self, file_path):
        """读取单个文件并累加到全局统计中"""
        state_dict = load_file(file_path)
        self.scanned_files.append(os.path.basename(file_path))
        
        for key, tensor in state_dict.items():
            # 记录 Key 和 Shape 用于展示
            shape_str = str(list(tensor.shape))
            self.global_keys.append((key, shape_str))
            
            # 统计参数量
            n = tensor.numel()
            self.global_total_params += n
            
            # 构建层级树 (取前两级作为 Key)
            parts = key.split('.')
            if len(parts) >= 2:
                prefix = f"{parts[0]}.{parts[1]}"
            else:
                prefix = parts[0]
            
            self.global_tree[prefix]["params"] += n
        
        del state_dict # 立即释放内存

    def print_final_report(self):
        # 1. 打印 Keys (如果开启)
        if self.show_all_keys:
            print(f"\n======== 完整 Module/Key 列表 ({len(self.global_keys)} 个) ========\n")
            # 排序
            self.global_keys.sort(key=lambda x: x[0])
            for key, shape in self.global_keys:
                print(f"{key:<70} | {shape}")
            print("\n" + "="*50)
        else:
            print(f"(提示: 使用 --show_keys 可查看 {len(self.global_keys)} 个 Key 的完整名称列表)")

        # 2. 总参数
        print(f"\n📊 模型总参数量: {format_size(self.global_total_params)}")
        
        # 3. 模块分布
        print("\n-------- 模块参数分布 (Top 2000) --------")
        sorted_tree = sorted(self.global_tree.items(), key=lambda x: x[1]['params'], reverse=True)
        
        if HAS_RICH:
            table = Table(title=f"Block Analysis (Total: {format_size(self.global_total_params)})")
            table.add_column("Block Name", style="cyan")
            table.add_column("Params", style="magenta")
            table.add_column("Ratio", style="yellow")
            
            for k, v in sorted_tree[:2000]: # 只看前20个大块
                p = v['params']
                ratio = (p / self.global_total_params) * 100
                table.add_row(k, format_size(p), f"{ratio:.1f}%")
            console.print(table)
        else:
            for k, v in sorted_tree[:20]:
                p = v['params']
                ratio = (p / self.global_total_params) * 100
                print(f"{k:<50} : {format_size(p)} ({ratio:.1f}%)")

    def print_config(self, model_path):
        # 简易 Config 读取
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
             config_path = os.path.join(model_path, "transformer", "config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                conf = json.load(f)
            print("\n--- 配置摘要 ---")
            keys = ["_class_name", "architectures", "num_attention_heads", "attention_head_dim", "in_channels", "patch_size", "num_layers"]
            for k in keys:
                if k in conf: print(f"{k}: {conf[k]}")

    def analyze_live_model(self, model):
        # 兼容旧逻辑：如果标准加载成功，直接分析 model 对象
        total = sum(p.numel() for p in model.parameters())
        print(f"📊 模型总参数量: {format_size(total)}")
        
        if self.show_all_keys:
            print("\n======== Module 列表 ========\n")
            for name, mod in model.named_modules():
                # 只打印叶子节点
                if len(list(mod.children())) == 0 and sum(p.numel() for p in mod.parameters()) > 0:
                     shapes = [str(list(p.shape)) for p in mod.parameters(recurse=False)]
                     print(f"{name:<60} | {', '.join(shapes)}")
                     
        # 也可以手动把 module 转换成 tree 来复用分布打印逻辑，这里简化处理直接打印
        print("\n(标准加载模式下，详细分布建议参考 named_modules 列表)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--show_keys", action="store_true")
    args = parser.parse_args()

    analyzer = ModelAnalyzer(show_all_keys=args.show_keys)
    
    if os.path.isfile(args.path):
        print("检测到单文件，分析中...")
        analyzer.process_single_file_content(args.path)
        analyzer.print_final_report()
    elif os.path.isdir(args.path):
        analyzer.analyze_diffusers(args.path)
    else:
        print(f"路径不存在: {args.path}")