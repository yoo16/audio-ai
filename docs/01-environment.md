# 01. 環境構築

## 対象環境

- WSL2 (Ubuntu) / Python 3.12
- GPU: RTX 5060 (Blackwell / sm_120, VRAM 8GB)
- PyTorch cu128 (CUDA 12.8) ※ Blackwell 世代は cu128 以上が必須
- パッケージ管理: uv

---

## 第1部: 文字起こし環境 (audio-ai 本体)

学習の仕組み理解を兼ねた前処理。実際の学習では GPT-SoVITS 側の ASR (0c) を使うため必須ではない。

### 依存の導入

```bash
cd ~/projects/audio-ai
uv add faster-whisper
uv add nvidia-cudnn-cu12          # cuDNN
uv add soundfile librosa         # WAV 変換用 (convert.py)
```

### CUDA ライブラリのパス通し

`faster-whisper` (ctranslate2) が cuDNN / cuBLAS を要求する。`stt.py` 冒頭で明示ロードしている:

```python
import os, ctypes
_nvidia_base = os.path.join(os.path.dirname(__file__),
    ".venv/lib/python3.12/site-packages/nvidia")
for _rel in ("cublas/lib/libcublas.so.12", "cudnn/lib/libcudnn.so.9",
             "cudnn/lib/libcudnn_ops.so.9", "cudnn/lib/libcudnn_cnn.so.9"):
    try:
        ctypes.CDLL(os.path.join(_nvidia_base, _rel), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
```

### 実行

```bash
source .venv/bin/activate
python stt.py       # audio/*.wav → transcriptions.list
python convert.py   # audio/ → audio_converted/ (16bit PCM / 32kHz / mono)
```

> エラーが出た場合は [05-troubleshooting.md](05-troubleshooting.md) を参照。

---

## 第2部: GPT-SoVITS 環境

### 1. クローン

```bash
cd ~/projects
git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
```

### 2. システム依存 (sudo)

```bash
sudo apt update && sudo apt install -y ffmpeg cmake build-essential
```

- `ffmpeg`: 音声デコード (load_audio が ffmpeg CLI を使う)
- `cmake` / `build-essential`: pyopenjtalk・opencc のビルドに必要

### 3. 専用 venv + PyTorch (RTX 5060 = cu128)

audio-ai とは依存 (numpy 等) が競合するため venv を分ける。

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu128
```

### 4. Python 依存

```bash
uv pip install -r extra-req.txt --no-deps   # faster-whisper
uv pip install -r requirements.txt          # pyopenjtalk, opencc 等をビルド
```

### 5. バージョン固定 (重要)

`requirements.txt` の指定だと新しすぎる版が入り不整合を起こすため、以下に固定する:

```bash
# WebUI の gradio 4.44 と互換を取る
uv pip install "fastapi==0.115.2" "starlette==0.40.0"

# VAD 用。onnxruntime-gpu は CUDA13 要求で読めないため CPU 版に
uv pip uninstall onnxruntime-gpu
uv pip install onnxruntime

# torchcodec が要求する NVIDIA NPP ライブラリ
uv pip install nvidia-npp-cu12
```

### 6. LD_LIBRARY_PATH (torchcodec 対策)

torchcodec は venv 内の nvidia 各ライブラリを実行時に探すため、WebUI 起動前に通す。
以下を通すと、そこから起動される学習・抽出サブプロセスもすべて継承する。

```bash
NV=$PWD/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib | tr '\n' ':')$LD_LIBRARY_PATH"
```

### 7. 事前学習モデルのダウンロード (~4.3GB)

HuggingFace `XXXXRT/GPT-SoVITS-Pretrained` から取得・展開。

```bash
source .venv/bin/activate
PY_PREFIX=$(python -c "import sys; print(sys.prefix)")
PYOPENJTALK_PREFIX=$(python -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))")
BASE="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"

wget -q "$BASE/pretrained_models.zip" -O pretrained_models.zip && unzip -q -o pretrained_models.zip -d GPT_SoVITS && rm pretrained_models.zip
wget -q "$BASE/G2PWModel.zip" -O G2PWModel.zip && unzip -q -o G2PWModel.zip -d GPT_SoVITS/text && rm G2PWModel.zip
wget -q "$BASE/nltk_data.zip" -O nltk_data.zip && unzip -q -o nltk_data.zip -d "$PY_PREFIX" && rm nltk_data.zip
wget -q "$BASE/open_jtalk_dic_utf_8-1.11.tar.gz" -O ojt.tar.gz && tar -xzf ojt.tar.gz -C "$PYOPENJTALK_PREFIX" && rm ojt.tar.gz
```

### 8. WebUI 起動

```bash
source .venv/bin/activate
NV=$PWD/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib | tr '\n' ':')$LD_LIBRARY_PATH"
python webui.py ja_JP
```

- ブラウザで **http://localhost:9874**
- `gio: ... Operation not supported` はブラウザ自動起動失敗の警告で無害
