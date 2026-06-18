#!/bin/bash
# ============================================================
# 饼哥帮你剪播客(Xianerbing-podcast-cutter) 启动脚本
# 双击此文件启动应用,浏览器会自动打开。
# 用完关掉这个窗口即可停止。
# ============================================================

cd "$(dirname "$0")"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}            饼哥帮你剪播客 启动中...                  ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo ""

# 检查是否安装过
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}  尚未安装!请先双击「安装PodcastZ.command」${NC}"
    echo ""
    read -n 1 -s -r -p "按任意键关闭..."
    exit 1
fi

source .venv/bin/activate

echo -e "${YELLOW}  正在启动服务,约5秒后浏览器会自动打开...${NC}"

# 后台延迟打开浏览器(给服务器3秒启动时间)
(
    sleep 3
    open "http://localhost:8000"
) &

# 启动服务器(前台运行,关闭窗口即停止)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

echo ""
echo -e "${YELLOW}  服务已停止。${NC}"
read -n 1 -s -r -p "按任意键关闭..."
