#!/bin/bash

# 获取脚本所在目录，确保路径正确
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# --- 颜色定义 ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}[INFO] 正在初始化运行环境...${NC}"

if [ -f "$DIR/python/bin/python" ]; then
    echo -e "${GREEN}[INFO] 检测到 Python 虚拟环境。${NC}"
else
    echo -e "${YELLOW}[INFO] 未检测到 Python 环境，正在通过系统 Python 创建虚拟环境...${NC}"
    
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}[ERROR] 未找到 python3。请先在您的 Linux 系统中安装 Python 3.10+。${NC}"
        echo "例如 (Ubuntu): sudo apt install python3 python3-venv"
        read -p "按回车键退出..."
        exit 1
    fi

    # 创建 venv
    python3 -m venv "$DIR/python"
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] 虚拟环境创建失败。请检查是否安装了 python3-venv。${NC}"
        read -p "按回车键退出..."
        exit 1
    fi
    echo -e "${GREEN}[INFO] Python 环境创建完成。${NC}"
fi

# 设置环境变量 PATH
export PATH="$DIR/python/bin:$PATH"


# --- Logo 显示 (保留原味) ---
echo "=========================================="
echo "      DiffPipe Forge 一键启动器 (Linux)"
echo "=========================================="
echo "ooooooooooooo  o8o            "
echo "            oooooooooo."
echo "8'   888   \`8  \`^\"'                        \`888'   \`Y8b                                   "
echo "     888      oooo   .oooo.   ooo. .oo.    888      "
echo "888  .ooooo.  ooo. .oo.    .oooooooo "
echo "     888      \`888  \`P  )88b  \`888P\"Y88b   888      888 d88' \`88b \`888P\"Y88b  888' \`88b  "
echo "     888       888   .oP\"888   888   888   888      888 888   888  888   888  888   888  "
echo "     888  "
echo "     888  d8(  888   888   888   888     d88' 888   888  888   888  \`88bod8P'  "
echo "    o888o     o888o \`Y888\"\"8o o888o o888o o888bood8P'   \`Y8bod8P' o888o o888o \`8oooooo.  "
echo "                                          "
echo "                                    d\"     YD  "
echo "                                                         "
echo "                     ^\"Y88888P'  "
echo "天冬AI制作：https://space.bilibili.com/32275117"
echo "=========================================="


# --- 检测 Node.js ---
echo -e "${GREEN}[INFO] 检查 Node.js 环境...${NC}"
if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v)
    echo -e "${GREEN}[INFO] Node.js 就绪: $NODE_VERSION${NC}"
else
    echo -e "${RED}[ERROR] 未检测到 Node.js。${NC}"
    echo "请手动安装 Node.js v22+ (推荐使用 nvm 或系统包管理器)。"
    echo "下载地址: https://nodejs.org/"
    read -p "按回车键退出..."
    exit 1
fi

# --- 启动应用 ---
# 进入 ui 目录
if [ -d "app/ui" ]; then
    cd app/ui
else
    echo -e "${RED}[ERROR] 找不到目录 app/ui，请确认脚本位置正确。${NC}"
    read -p "按回车键退出..."
    exit 1
fi

# 检查依赖
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}[INFO] 检测到依赖缺失，正在安装...${NC}"
    npm install
fi

echo -e "${GREEN}[INFO] 正在启动开发服务器...${NC}"

# 启动循环 (包含重试逻辑)
RETRY_COUNT=0

while true; do
    npm run dev
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        break
    fi

    if [ "$RETRY_COUNT" == "1" ]; then
        break
    fi

    echo -e "${YELLOW}[WARNING] 服务器启动异常，正在尝试自动修复...${NC}"
    echo -e "${GREEN}[INFO] 正在重新安装依赖 (npm install)...${NC}"
    npm install
    RETRY_COUNT=1
    echo -e "${GREEN}[INFO] 正在重试启动...${NC}"
done

echo
read -p "程序已结束，按回车键关闭窗口..."