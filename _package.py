"""饼哥帮你剪播客(Xianerbing-podcast-cutter) 打包脚本(UTF-8 安全版)。

用法:python _package.py
或:双击 package.bat(它会调用本脚本)

修复了 PowerShell Compress-Archive 对中文文件名编码的问题:
- 文件名用 UTF-8 标志位(0x800),Mac 解压不乱码
- 文件内容字节原样写入,不做编码转换
"""
import os
import shutil
import zipfile
from pathlib import Path

SRC = Path(__file__).resolve().parent
DIST = SRC / "dist"
ZIP = DIST / "Xianerbing-podcast-cutter-v0.1.0-mac.zip"

EXCLUDE_DIRS = {".venv", "models", "data", "dist", "__pycache__", ".pytest_cache",
                ".ruff_cache", ".git", ".idea", ".vscode"}
EXCLUDE_FILES = {"package.bat", "_package.py", ".env", ".env.local", "127"}
EXCLUDE_SUFFIX = {".wav", ".mp3", ".m4a", ".log", ".pyc"}


def is_temp(p: Path) -> bool:
    return p.name.startswith("_")


print(f"[1/2] Collecting from {SRC}")
DIST.mkdir(parents=True, exist_ok=True)
if ZIP.exists():
    ZIP.unlink()

file_count = 0
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for fn in files:
            fp = Path(root) / fn
            rel = fp.relative_to(SRC)
            if fn in EXCLUDE_FILES:
                continue
            if fp.suffix.lower() in EXCLUDE_SUFFIX:
                continue
            if is_temp(fp):
                continue
            arcname = str(rel).replace("\\", "/")
            zf.write(fp, arcname)
            file_count += 1

    # 占位目录
    zf.writestr("data/.gitkeep", "")
    zf.writestr("models/.gitkeep", "")

    # Do not package .env. It may contain private API keys.
    # The installer creates a local .env from .env.example when needed.

size_mb = ZIP.stat().st_size / (1024 * 1024)
print(f"\nDone: {ZIP}")
print(f"Size: {size_mb:.1f} MB, {file_count} files")
