
import subprocess
import sys
import re
import os
import shutil


REQUIRED_PACKAGES = {
    "deepspeed",
    "bitsandbytes", 
    "flash-attn",
    
    "toml",
    "transformers",
    "diffusers",
    "datasets",
    "pillow",
    "sentencepiece",
    "protobuf",
    "peft",
    "torch-optimi",
    "tensorboard",
    "tqdm",
    "safetensors",
    "imageio",  
    "av",
    "einops",
    "accelerate",
    "loguru",
    "omegaconf",
    "iopath",
    "termcolor",
    "hydra-core",
    "easydict",
    "ftfy",
    "pytorch-optimizer",
    "wandb",
    "optimum-quanto",
    "psutil",
    "mpi4py",
    "wheel",
    "ninja",
    "pynvml",
    
    "torch",
    "torchvision", 
    "torchaudio",
    
    "numpy",
    "scipy",
    "pip",
    "setuptools",
    
    "triton",
    "xformers",
    "sageattention",
    "natten",
    "nunchaku",
}

PACKAGES_TO_REMOVE = [
    "comfyui-embedded-docs",
    "comfyui_frontend_package",
    "comfyui_workflow_templates",
    "comfyui-workflow-templates-core",
    "comfyui-workflow-templates-media-api",
    "comfyui-workflow-templates-media-image",
    "comfyui-workflow-templates-media-other",
    "comfyui-workflow-templates-media-video",
    
    # LLM 框架
    "langchain",
    "langchain-core",
    "langchain-community",
    "langchain-ollama",
    "langchain-openai",
    "langchain-text-splitters",
    "llama-index",
    "llama-index-cli",
    "llama-index-core",
    "llama-index-embeddings-openai",
    "llama-index-indices-managed-llama-cloud",
    "llama-index-instrumentation",
    "llama-index-llms-openai",
    "llama-index-readers-file",
    "llama-index-readers-llama-parse",
    "llama-index-workflows",
    "llama-cloud",
    "llama-cloud-services",
    "llama_cpp_python",
    "llama-parse",
    "openai",
    "anthropic",
    "cohere",
    "litellm",
    "litelama",
    "mistralai",
    "cerebras_cloud_sdk",
    "zhipuai",
    "dashscope",
    "replicate",
    "ollama",
    "aisuite",
    
    # CrewAI
    "crewai",
    
    # GUI 框架
    "streamlit",
    "gradio",
    "PyQt5",
    "PyQt5-Qt5",
    "PyQt5_sip",
    
    # Web 框架 (如果不需要)
    # "fastapi",  # 保留，可能被某些包依赖
    # "uvicorn",
    
    # 图像处理的额外库（如果不需要）
    "mediapipe",
    "insightface",
    "deepface",
    "mtcnn",
    "realesrgan",
    
    # 音频处理（如果不需要）
    "librosa",
    "soundfile",
    "sounddevice",
    "faster-whisper",
    "stanza",
    
    # 中文处理
    "OpenCC",
    "cn2an",
    "WeTextProcessing",
    
    # 知识图谱
    "neo4j",
    "chromadb",
    "chroma-hnswlib",
    
    # 云服务SDK
    "boto3",
    "botocore",
    "awscli",
    "azure-core",
    "azure-storage-blob",
    "aliyun-python-sdk-core",
    "aliyun-python-sdk-kms",
    "oss2",
    "s3transfer",
    
    # 翻译
    "argostranslate",
    "deep-translator",
    
    # 其他大型库
    "tensorflow-estimator",
    "tensorflow-io-gcs-filesystem",
    "tf_keras",
    "selenium",
    "playwright",
    "pyautogui",
    
    # 数据库
    "redis",
    "aiosqlite",
    "SQLAlchemy",
    
    # 杂项
    "wikipedia",
    "arxiv",
    "civitai-py",
    "roboflow",
    "supervision",
    "ultralytics",
    "ultralytics-thop",
    "ultralyticsplus",
    "yolov5",
    
    # MCP 相关
    "mcp",
    
    # 监控
    "prometheus_client",
    "prometheus-fastapi-instrumentator",
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-proto",
    "opentelemetry-exporter-otlp-proto-common",
    "opentelemetry-exporter-otlp-proto-grpc",
    "opentelemetry-exporter-otlp-proto-http",
    "opentelemetry-instrumentation",
    "opentelemetry-instrumentation-asgi",
    "opentelemetry-instrumentation-fastapi",
    "opentelemetry-semantic-conventions",
    "opentelemetry-util-http",
    "sentry-sdk",
    "posthog",
    
    # 测试框架
    "pytest",
    "pytest-asyncio",
    "tox",
    
    # NLP (如果不需要)
    "spacy",
    "spacy-legacy",
    "spacy-loggers",
    "nltk",
    "textblob",
    
    # Web scraping
    "beautifulsoup4",
    
    # PDF 处理
    "pdfminer.six",
    "pdfplumber",
    "pypdf",
    "pypdfium2",
    "PyMuPDF",
    "reportlab",
    
    # Office 文档
    "docx2txt",
    "openpyxl",
    "xlrd",
    
    # 3D 相关 (如果不需要)
    "trimesh",
    "pymeshlab",
    "manifold3d",
    "pycollada",
    "pygltflib",
    "xatlas",
    "vhacdx",
    "PyMCubes",
    "nerfacc",
    
    # Discord/Matrix bots
    "py-cord",
    "matrix-client",
    "matrix-nio",
    
    # 其他
    "skypilot",
    "docker",
    "cookiecutter",
]


def get_installed_packages():
    """获取已安装的包列表"""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        capture_output=True,
        text=True
    )
    packages = {}
    for line in result.stdout.strip().split('\n'):
        if '==' in line:
            name, version = line.split('==', 1)
            packages[name.lower()] = version
    return packages


def uninstall_packages(packages):
    """卸载指定的包"""
    if not packages:
        print("没有需要卸载的包")
        return
    
    print(f"\n准备卸载 {len(packages)} 个包:")
    for pkg in sorted(packages)[:20]:
        print(f"  - {pkg}")
    if len(packages) > 20:
        print(f"  ... 还有 {len(packages) - 20} 个包")
    
    confirm = input("\n确认卸载？(y/n): ")
    if confirm.lower() != 'y':
        print("已取消")
        return
    
    # 批量卸载
    for pkg in packages:
        print(f"卸载: {pkg}")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
            capture_output=True
        )
    
    print("\n卸载完成！")


def main():
    print("=" * 50)
    print("Python 环境瘦身工具")
    print("=" * 50)
    
    # 获取 Python 环境路径
    python_dir = os.path.dirname(sys.executable)
    
    print(f"\nPython 环境路径: {python_dir}")
    
    installed = get_installed_packages()
    print(f"当前已安装: {len(installed)} 个包")
    
    # 找出要删除的包
    to_remove = []
    for pkg in PACKAGES_TO_REMOVE:
        pkg_lower = pkg.lower()
        if pkg_lower in installed:
            to_remove.append(pkg)
    
    print(f"可以删除的包: {len(to_remove)} 个")
    
    print("\n请选择操作:")
    print("[1] 删除不需要的包")
    print("[2] 清理缓存文件")
    print("[3] 全部执行 (删包 + 清缓存)")
    print("[0] 退出")
    
    choice = input("\n请输入选项: ")
    
    if choice == "1":
        if to_remove:
            uninstall_packages(to_remove)
        else:
            print("没有找到可以删除的包")
    elif choice == "2":
        clean_cache(python_dir)
    elif choice == "3":
        if to_remove:
            uninstall_packages(to_remove)
        clean_cache(python_dir)
    else:
        print("已退出")


def clean_cache(python_dir):
    """清理缓存文件"""
    print("\n" + "=" * 50)
    print("清理缓存")
    print("=" * 50)
    
    total_size = 0
    deleted_count = 0
    
    # 1. 清理 pip 缓存
    print("\n[1/4] 清理 pip 缓存...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "cache", "purge"],
        capture_output=True,
        text=True
    )
    print("  pip 缓存已清理")
    
    # 2. 清理 __pycache__ 目录
    print("\n[2/4] 清理 __pycache__ 目录...")
    for root, dirs, files in os.walk(python_dir):
        for dir_name in dirs[:]:  # 使用切片避免修改迭代中的列表
            if dir_name == "__pycache__":
                cache_path = os.path.join(root, dir_name)
                try:
                    size = get_dir_size(cache_path)
                    shutil.rmtree(cache_path)
                    total_size += size
                    deleted_count += 1
                except Exception as e:
                    print(f"  无法删除: {cache_path} - {e}")
    print(f"  删除了 {deleted_count} 个 __pycache__ 目录")
    
    # 3. 清理 .pyc 文件
    print("\n[3/4] 清理 .pyc 文件...")
    pyc_count = 0
    for root, dirs, files in os.walk(python_dir):
        for file in files:
            if file.endswith('.pyc'):
                pyc_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(pyc_path)
                    os.remove(pyc_path)
                    total_size += size
                    pyc_count += 1
                except Exception as e:
                    pass
    print(f"  删除了 {pyc_count} 个 .pyc 文件")
    
    # 4. 清理 *.dist-info 中的冗余文件 (RECORD, INSTALLER 等)
    print("\n[4/4] 清理包元数据中的冗余文件...")
    lib_path = os.path.join(python_dir, "Lib", "site-packages")
    if os.path.exists(lib_path):
        redundant_files = ["RECORD", "INSTALLER", "REQUESTED", "direct_url.json"]
        redundant_count = 0
        for item in os.listdir(lib_path):
            if item.endswith(".dist-info"):
                dist_info_path = os.path.join(lib_path, item)
                for rf in redundant_files:
                    rf_path = os.path.join(dist_info_path, rf)
                    if os.path.exists(rf_path):
                        try:
                            size = os.path.getsize(rf_path)
                            os.remove(rf_path)
                            total_size += size
                            redundant_count += 1
                        except:
                            pass
        print(f"  删除了 {redundant_count} 个冗余元数据文件")
    
    print(f"\n✅ 缓存清理完成！共释放 {format_size(total_size)} 空间")


def get_dir_size(path):
    """获取目录大小"""
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except:
                pass
    return total


def format_size(size):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


if __name__ == "__main__":
    main()

