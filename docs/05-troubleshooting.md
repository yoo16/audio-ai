# 05. トラブルシューティング (ハマりどころ集)

実際に遭遇したエラーと対処をまとめる。多くは **PyTorch 2.11 (cu128) が新しすぎる**ことに起因。

---

## 文字起こし (stt.py / faster-whisper)

### `libcudnn_ops.so.9 ... cannot load` / `cudnnCreateTensorDescriptor`
- 原因: cuDNN 未導入
- 対処: `uv add nvidia-cudnn-cu12`

### `Library libcublas.so.12 is not found`
- 原因: cuBLAS のパスが通っていない
- 対処: `stt.py` 冒頭で ctypes により明示ロード ([01](01-environment.md) 参照)
- 補足: `os.environ["LD_LIBRARY_PATH"]` を Python 実行後に設定しても
  ローダーには効かないため、ctypes ロードか起動前 export が必要

### `Requested int8_float16 / int8 ... not supported`
- 原因: その compute_type が非対応
- 対処: `compute_type="float16"` を使う (cuDNN/cuBLAS を通せば動く)

### `HF_TOKEN` 警告
- 無視可 (レート制限のみ)。必要なら `export HF_TOKEN=...`

---

## GPT-SoVITS 環境

### WebUI で `TypeError: unhashable type: 'dict'` (jinja2)
- 原因: `requirements.txt` の `fastapi[standard]>=0.115.2` が
  fastapi 0.139 / starlette 1.x を引き込み、gradio 4.44 と非互換
- 対処: `uv pip install "fastapi==0.115.2" "starlette==0.40.0"`

### ASR (0c) の list が空になる / `Applying the VAD filter requires the onnxruntime package`
- 原因: onnxruntime-gpu が `libcudart.so.13` (CUDA13) を要求し読めない
- 対処: CPU 版に差し替え
  ```bash
  uv pip uninstall onnxruntime-gpu
  uv pip install onnxruntime
  ```
- 差し替え後は WebUI 再起動が必要

### `torchcodec ... libtorchcodec_core4.so could not load` → `libnppicc.so.12: cannot open shared object file`
- 原因: torchcodec が NVIDIA NPP ライブラリを見つけられない
- 対処:
  ```bash
  uv pip install nvidia-npp-cu12
  # かつ起動前に LD_LIBRARY_PATH を通す
  NV=$PWD/.venv/lib/python3.12/site-packages/nvidia
  export LD_LIBRARY_PATH="$(ls -d $NV/*/lib | tr '\n' ':')$LD_LIBRARY_PATH"
  ```
- 補足: WebUI をこの環境で起動すれば、学習・抽出サブプロセスも継承する
- 注意: SoVITS 学習の音声読み込み (`load_audio`) は ffmpeg CLI を使うため
  torchcodec 非依存。torchcodec が要るのは推論・SV 抽出・エクスポート系

---

## 学習

### SoVITS 学習が `ZeroDivisionError: division by zero` (data_utils.py)
- 原因: v2Pro 用の話者特徴 `logs/yoo/7-sv_cn` が空。
  音素・SSL・wav・SV の 4 つの積集合が空になり `leng=0`
- 対処: SV 抽出 (`2-get-sv.py`) を実行して 7-sv_cn を生成 ([02](02-training.md) 参照)
- 教訓: **v2Pro はワンクリック三連に SV 抽出が含まれない**。別途実行が必要

### GPT 学習が `UnpicklingError: Weights only load failed ... GLOBAL pathlib.PosixPath`
- 原因: torch>=2.6 で `torch.load` の `weights_only` デフォルトが True になり、
  pytorch_lightning が読む ckpt 内の `pathlib.PosixPath` が拒否される
- 対処: `GPT_SoVITS/s1_train.py` の `import torch` 直後にパッチ
  ```python
  _orig_torch_load = torch.load
  def _patched_torch_load(*args, **kwargs):
      kwargs["weights_only"] = False  # lightning が明示 True を渡すので強制上書き
      return _orig_torch_load(*args, **kwargs)
  torch.load = _patched_torch_load
  ```
- 注意: `kwargs.setdefault(...)` では効かない。lightning が `weights_only=True` を
  明示的に渡すため、キー代入で**強制上書き**する必要がある

---

## プロセス操作の注意

- `pkill -f "webui.py"` は**自分のシェルのコマンドラインにもマッチして自滅**することがある
  (exit 144)。PID 指定 (`kill <pid>`) や、より限定的なパターンを使う
- WebUI 再起動時はポート (9874 / 9871 / 9872) が解放されているか
  `curl` で確認してから起動する
