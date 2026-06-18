"""PodcastZ FastAPI 主入口。

启动:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

打开浏览器 http://localhost:8000 即可使用。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.api.routes import router as api_router
from backend.config import settings, FRONTEND_DIR
from backend.ffmpeg_util import ensure_ffmpeg


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动检查
    try:
        ensure_ffmpeg()
        logger.info("ffmpeg 检查通过")
    except Exception as e:  # noqa: BLE001
        logger.error(f"ffmpeg 不可用: {e}")
    logger.info(f"PodcastZ 启动 → http://{settings.host}:{settings.port}")
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="中文播客口癖/填充词智能剪辑工具",
    lifespan=lifespan,
)

# CORS(本地开发前端可能跨端口;服务器部署也可置于反代后)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API
app.include_router(api_router)

# 静态前端
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """首页 —— 加载前端。"""
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return HTMLResponse("<h1>PodcastZ</h1><p>前端目录缺失,仅 API 可用: /docs</p>")


@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse("", status_code=204)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True)
