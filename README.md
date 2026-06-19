# 🎙️ 饼哥帮你剪播客 — Xianerbing-podcast-cutter

由用户上传音频,自动剪掉口癖、句子中的无效填充词,**严格保留所有有意义的词语**,
并通过多项平滑技术确保**剪辑后不突兀**。本地开箱即用,可一键部署到服务器。

> 设计目标三件套:① 解决"剪辑后突兀";② 严格按中文语义;③ 本地可用 → 服务器可部署。

---

## ✨ 核心特性

### 1. 解决"剪辑后突兀"(重中之重)
针对拼接处最容易出问题的几个点,全部做了平滑处理(`backend/pipeline/smooth.py`):

| 技术 | 作用 |
|------|------|
| **零交叉吸附** | 每个剪切边界对齐最近零交叉点,消除咔哒声/爆音 |
| **节奏保持** | 句间口癖删后用房间底噪填充,长度=说话人典型停顿,保留呼吸节奏 |
| **等功率交叉淡化** | 拼接点 15–30ms equal-power crossfade(cos² 曲线),消除音量跳变 |
| **底噪一致性** | 从录音最安静的 500ms 采样房间底噪填充删除区,保证环境感连续 |
| **呼吸声保留** | VAD 区分纯静音 vs 换气声,呼吸是自然的,不删 |
| **软限幅防爆音** | tanh 软限幅,防止拼接瞬时过冲 |

### 2. 严格按中文语义设计
不是粗暴字典匹配。每个歧义词走**语义守卫**(`semantic.py`):

- **"然后"** → 接动作且前句完整 = 连接词(保留);独立/重复/拖音 = 口癖(删)
- **"就是"** → 接"是 X"判断/定义(保留);接"说/那个"(删)
- **"那个/这个"** → 接名词作指示(保留);拖音/句末(删)
- **"那么"** → 接推导承接(保留);拖音(删)
- **"其实/基本上/反正"** → 接陈述(保留);接话语标记链(删)
- **重复词** → "我 我 我想说"保留首个,删其余
- 规则无法裁决的少数 → 交 **LLM 复核**(可选,带上下文,失败保守保留)

### 3. 双审核流程
- **人工审核(默认)**:输出剪辑清单(时间戳+原文+原因+置信度),波形高亮,可逐条试听/勾选/撤销
- **全自动模式**:上传即出片,附可查清单
- **力度滑块**:保守 / 均衡 / 激进,实时切换

---

## 🚀 快速开始(本地)

### 前置
- Python 3.10+(已在 3.14 验证)
- ffmpeg 会通过 `imageio-ffmpeg` 自动打包,**无需手动安装**

### 安装与运行
```bash
cd E:\ZProject\PodcastZ

# 1) 创建虚拟环境
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# 2) 安装依赖
pip install -e .

# 3) 配置(可选,不配也能跑 —— 纯离线规则)
cp .env.example .env
#   - WHISPER_MODEL=large-v3, WHISPER_DEVICE=cuda, WHISPER_COMPUTE=int8_float16 (GPU 精准模式)
#   - LLM_* (可选,填了启用大模型复核)

# 4) 启动
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 5) 打开浏览器
#    http://localhost:8000
```

首次转写会下载 Whisper 模型(`large-v3`≈3GB),缓存到 `models/`。有 NVIDIA GPU 时会走 CUDA;无 GPU 可把 `.env` 改回 `WHISPER_MODEL=medium`、`WHISPER_DEVICE=cpu`、`WHISPER_COMPUTE=int8`。

---

## 🐳 服务器部署(Docker)

```bash
cp .env.example .env       # 按需修改
docker compose up -d --build
# 访问 http://<服务器IP>:8000
```

GPU 加速:编辑 `docker-compose.yml` 取消 `deploy.resources` 注释,需服务器装 `nvidia-container-toolkit`。

---

## 🧠 工作流程

```
上传音频
   ↓
[转写] faster-whisper (zh, 词级时间戳, VAD 去静音)
   ↓
[VAD]   能量+过零率 → 停顿 / 呼吸 区间
   ↓
[检测]  口癖词典 + 词性 + 规则 → 候选 EditItem
   ↓
[守卫]  语义判定(然后/就是/那个… 是连接词还是口癖?)
   ↓ (可选)
[LLM]   歧义项批量复核(OpenAI 兼容)
   ↓
[审核]  前端:逐条试听/勾选/撤销
   ↓
[方案]  生成删除区间(CutRegion)
   ↓
[平滑]  零交叉 + 交叉淡化 + 底噪填充 + 节奏保持   ← 核心
   ↓
[渲染]  MP3 + EBU R128 两遍 LUFS 归一(-16 LUFS)
   ↓
下载
```

---

## ⚙️ 关键配置(`.env`)

| 配置 | 默认 | 说明 |
|------|------|------|
| `WHISPER_MODEL` | `large-v3` | tiny/base/small/medium/large-v3 |
| `WHISPER_DEVICE` | `auto` | auto/cpu/cuda; auto 会优先用 CTranslate2 检测到的 CUDA |
| `WHISPER_COMPUTE` | `int8_float16` | GPU 推荐 int8_float16; CPU 会自动使用 int8 |
| `DEFAULT_STRENGTH` | `balanced` | conservative/balanced/aggressive |
| `CROSSFADE_MS` | `25` | 等功率交叉淡化长度 |
| `FILL_FLOOR_NOISE` | `true` | 用房间底噪填充(否则纯静音) |
| `TARGET_PAUSE_MS` | `280` | 节奏保持的目标停顿长度 |
| `LOUDNORM_TARGET_LUFS` | `-16.0` | EBU R128 播客标准 |
| `LLM_BASE_URL` | (空) | OpenAI 兼容,如 `https://api.deepseek.com/v1` |
| `LLM_API_KEY` | (空) | 不填则纯离线规则 |

---

## 📁 项目结构

```
Xianerbing-podcast-cutter/
├── backend/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 全局配置
│   ├── models.py            # 数据模型(EditItem 等)
│   ├── ffmpeg_util.py       # ffmpeg 二进制解析(imageio-ffmpeg)
│   ├── api/routes.py        # REST API
│   ├── pipeline/
│   │   ├── transcribe.py    # Whisper 词级时间戳
│   │   ├── vad.py           # 停顿/呼吸检测
│   │   ├── lexicon.py       # 中文口癖词典 + 力度策略
│   │   ├── detect.py        # 检测主模块
│   │   ├── semantic.py      # 语义守卫 ⭐
│   │   ├── semantic_llm.py  # LLM 复核(可选)
│   │   ├── editplan.py      # 剪辑方案
│   │   ├── smooth.py        # 平滑剪辑 ⭐⭐⭐(核心)
│   │   └── render.py        # 渲染 + LUFS 归一
│   └── workers/task_queue.py
├── frontend/                # 单页原生 HTML/JS
├── Dockerfile / docker-compose.yml
├── pyproject.toml / .env.example
└── README.md
```

---

## ❓ 常见问题

**Q: 首次启动很慢?**
A: 在下载 Whisper large-v3 模型(约3GB),之后缓存复用。

**Q: 没有 GPU 能用吗?**
A: 能。把 `.env` 改为 `WHISPER_MODEL=medium`、`WHISPER_DEVICE=cpu`、`WHISPER_COMPUTE=int8` 即可;GPU 模式默认用 `large-v3` 追求更高转录精度。

**Q: 不填 LLM key 会怎样?**
A: 完全可用。歧义项按力度保守处理(均衡力度下默认保留待人工审核),功能不残缺。

**Q: 会不会误删有意义的词?**
A: 三重保险:① 语义守卫规则;② 默认 keep=True(用户审核);③ 力度阈值。
   即使误判,人工审核模式可在波形上逐条撤销。

---

## 📜 许可
MIT
