# 自分の声の音声合成モデル作成ガイド (GPT-SoVITS)

自分の音声から、任意テキストを自分の声で読み上げる TTS モデルを作る手順。
文字起こし (faster-whisper) → GPT-SoVITS 学習の流れをまとめる。

## 環境

- WSL2 (Ubuntu) / Python 3.12
- GPU: RTX 5060 (Blackwell / sm_120, VRAM 8GB)
- CUDA: PyTorch cu128 (CUDA 12.8)
- パッケージ管理: uv

---

## 第1部: 文字起こし (audio-ai プロジェクト)

学習の仕組み理解を兼ねた前処理。実際の学習では GPT-SoVITS 側の ASR を使うので必須ではない。

### 1. 音声を用意

`audio/*.wav` に自分の音声を配置（今回は 16 ファイル / 合計約 1.7 分）。

### 2. faster-whisper で文字起こし (`stt.py`)

```bash
source .venv/bin/activate
python stt.py
```

- モデル: `large-v3-turbo` / `device="cuda"` / `compute_type="float16"`
- 出力: `transcriptions.list`（GPT-SoVITS形式 `パス|話者|言語|テキスト`）

#### ハマりどころと対処

| エラー | 原因 | 対処 |
|---|---|---|
| `libcudnn_ops.so.9 ... cannot load` | cuDNN 未導入 | `uv add nvidia-cudnn-cu12` |
| `libcublas.so.12 is not found` | cuBLAS のパス未通し | `stt.py` 冒頭で ctypes 直接ロード（下記） |
| `HF_TOKEN` 警告 | 未認証DL | 無視可（レート制限のみ） |

`stt.py` 冒頭で CUDA ライブラリを明示ロード:

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

### 3. WAV フォーマット変換 (`convert.py`)

元音声が 32bit float だったため、GPT-SoVITS 用に **16bit PCM / 32kHz / モノラル**へ変換。

```bash
uv add soundfile librosa
python convert.py   # audio/ → audio_converted/
```

---

## 第2部: GPT-SoVITS の導入

### 1. クローン

```bash
cd ~/projects
git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git
```

### 2. システム依存

```bash
sudo apt update && sudo apt install -y ffmpeg cmake build-essential
```

### 3. 専用 venv + PyTorch (RTX 5060 = cu128)

```bash
cd ~/projects/GPT-SoVITS
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu128
```

### 4. Python 依存

```bash
uv pip install -r extra-req.txt --no-deps   # faster-whisper
uv pip install -r requirements.txt          # pyopenjtalk, opencc 等をビルド
```

#### ハマりどころと対処

| 症状 | 原因 | 対処 |
|---|---|---|
| WebUI で `unhashable type: 'dict'` | fastapi/starlette が新しすぎて gradio 4.44 と非互換 | `uv pip install "fastapi==0.115.2" "starlette==0.40.0"` |
| ASR の list が空 / `VAD requires onnxruntime` | onnxruntime-gpu が CUDA13 要求で読めない | `uv pip uninstall onnxruntime-gpu` → `uv pip install onnxruntime`（CPU版） |

### 5. 事前学習モデルのダウンロード (~4.3GB)

HuggingFace `XXXXRT/GPT-SoVITS-Pretrained` から取得・展開:

- `pretrained_models.zip` → `GPT_SoVITS/`
- `G2PWModel.zip` → `GPT_SoVITS/text/`
- `nltk_data.zip` → venv prefix
- `open_jtalk_dic_utf_8-1.11.tar.gz` → pyopenjtalk パッケージ内

### 6. WebUI 起動

```bash
source .venv/bin/activate
python webui.py ja_JP
```

- ブラウザで **http://localhost:9874**
- `gio: ... Operation not supported` はブラウザ自動起動失敗の警告で無害

---

## 第3部: WebUI での学習

### 前処理タブ

| ツール | 内容 | 今回 |
|---|---|---|
| 0a UVR5 | BGM/残響除去 | スキップ（BGM無し） |
| 0b 音声分割 | 数秒単位にスライス | 実行（16→19セグメント） |
| 0c 音声認識 | Faster Whisper で文字起こし → `.list` | 実行 |
| 0d テキスト校正 | 誤認識を目視修正 | 実行 |

- 0b 入力: `/home/yoo/projects/audio-ai/audio_converted` → 出力 `output/slicer_opt`
- 0c 入力: `output/slicer_opt` → 出力 `output/asr_opt/slicer_opt.list`（モデル large-v3 / 言語 ja / float16）
- 0d: `作戦→作成` などの誤認識を修正

### GPT-SoVITS TTS タブ

**設定**
- 実験/モデル名: `yoo`
- 学習済みモデルバージョン: `v2Pro`

**1A-データセット整形**（ワンクリック三連 or 個別に 1Aa→1Ab→1Ac）
- listパス: `output/asr_opt/slicer_opt.list`
- 音声フォルダ: `output/slicer_opt`
- 生成物: `logs/yoo/` に `2-name2text.txt` `3-bert` `4-cnhubert` `5-wav32k` `6-name2semantic.tsv` `7-sv_cn`

**1B-微調整学習**（RTX 5060 / 8GB 向け）
- SoVITS学習: batch_size `2〜3`（OOM時は `1`）/ total_epoch `8〜15` / fp16 ON
- GPT学習: 同様に小さめの batch_size

**1C-推論**
- 学習したモデルを選択 → 参照音声 + テキスト → 自分の声で生成

---

## データ量の目安

| 音声量 | 品質 |
|---|---|
| ~2分（今回） | 試作レベル。声質は似るが不安定 |
| 5〜10分 | 実用的 |
| 30分〜1時間 | 高品質・安定 |

まず少量で試作 → 良ければ音声を追加して再学習、が現実的。

---

## 進捗チェックリスト

- [x] stt.py で文字起こし
- [x] convert.py で 16bit PCM / 32kHz へ変換
- [x] GPT-SoVITS クローン・依存導入・事前学習モデルDL
- [x] WebUI 起動
- [x] 0b 分割 / 0c ASR / 0d 校正
- [x] 1A データセット整形（音素・BERT・HuBERT・セマンティック）
- [ ] 1B SoVITS学習
- [ ] 1B GPT学習
- [ ] 1C 推論で音声生成
