#!/bin/bash
# ============================================================
# 饼哥帮你剪播客(Xianerbing-podcast-cutter) 一键安装脚本(Mac, Apple Silicon)
# 同事双击此文件即可,无需打开终端、无需输命令。
# ============================================================
# 此文件是 .command,Mac 上双击会用"终端"运行。
# 同事会看到一个进度窗口,等它跑完(约10分钟)即可。

set -e  # 任何命令失败就停止

# 切到脚本所在目录(项目根)
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

# 彩色输出
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}   饼哥帮你剪播客 安装程序(首次运行约10分钟)        ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo ""
echo "请在安装过程中保持窗口打开,不要关闭。"
echo ""

# ============================================================
# 步骤 1:检测 Python
# ============================================================
echo -e "${YELLOW}[1/5] 检测 Python...${NC}"

PYTHON=""
if command -v python3 &> /dev/null; then
    PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0")
    PYOK=$(python3 -c 'import sys; print(1 if sys.version_info >= (3,10) else 0)' 2>/dev/null || echo "0")
    if [ "$PYOK" = "1" ]; then
        PYTHON=$(command -v python3)
        echo -e "${GREEN}  ✓ 已找到 Python3 ($PYVER)${NC}"
    fi
fi

if [ -z "$PYTHON" ]; then
    echo -e "${YELLOW}  未找到合适的 Python,尝试自动安装...${NC}"
    # 检测 Homebrew
    if ! command -v brew &> /dev/null; then
        echo -e "${YELLOW}  安装 Homebrew(苹果的包管理器)...${NC}"
        echo "  可能会要求输入密码(就是开机密码,输入时屏幕不显示是正常的)"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # 加入 PATH
        if [ -f /opt/homebrew/bin/brew ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [ -f /usr/local/bin/brew ]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi
    echo -e "${YELLOW}  用 Homebrew 安装 Python...${NC}"
    brew install python@3.12
    PYTHON=$(command -v python3.12 || command -v python3)
    echo -e "${GREEN}  ✓ Python 安装完成${NC}"
fi

# ============================================================
# 步骤 2:创建虚拟环境
# ============================================================
echo ""
echo -e "${YELLOW}[2/5] 创建运行环境...${NC}"
if [ -d ".venv" ]; then
    echo "  已存在,跳过"
else
    "$PYTHON" -m venv .venv
    echo -e "${GREEN}  ✓ 运行环境已创建${NC}"
fi
source .venv/bin/activate

# ============================================================
# 步骤 3:安装依赖
# ============================================================
echo ""
echo -e "${YELLOW}[3/5] 安装依赖包(约5分钟,请耐心等待)...${NC}"
python -m pip install --upgrade pip --quiet
# 安装项目依赖(分批装,避免单次超时)
pip install --quiet numpy scipy soundfile pydub ffmpeg-python imageio-ffmpeg loguru tqdm
pip install --quiet "fastapi>=0.110" "uvicorn[standard]" python-multipart "pydantic>=2.6" pydantic-settings
pip install --quiet jieba "opencc-python-reimplemented" openai httpx python-dotenv
pip install --quiet "faster-whisper>=1.0.3"
echo -e "${GREEN}  ✓ 依赖安装完成${NC}"

# ============================================================
# 步骤 4:下载 Whisper 模型(约1.5GB,走国内镜像)
# ============================================================
echo ""
echo -e "${YELLOW}[4/5] 下载语音识别模型(约1.5GB,首次约5-10分钟)...${NC}"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_SYMLINKS_WARNING="1"
python -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
from pathlib import Path
p = snapshot_download(
    'Systran/faster-whisper-medium',
    allow_patterns=['*.json', '*.txt', 'model.bin', 'tokenizer*', 'vocabulary*', 'preprocessor*'],
    cache_dir=str(Path('models')),
    max_workers=2,
)
print('模型路径:', p)
"
echo -e "${GREEN}  ✓ 模型下载完成${NC}"

# ============================================================
# 步骤 5:配置检查
# ============================================================
echo ""
echo -e "${YELLOW}[5/5] 检查配置...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}  已从模板创建 .env(默认配置,直接可用)${NC}"
else
    echo -e "${GREEN}  ✓ 已有 .env 配置${NC}"
fi

# 检查 ffmpeg
python -c "from backend.ffmpeg_util import ffmpeg_bin; print('  ffmpeg:', ffmpeg_bin())"

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ 安装完成!${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo ""
echo "  接下来:"
echo -e "  双击 ${GREEN}启动PodcastZ.command${NC} 开始使用饼哥帮你剪播客"
echo ""
echo "  (此窗口现在可以关闭了)"
echo ""
read -n 1 -s -r -p "按任意键关闭此窗口..."
