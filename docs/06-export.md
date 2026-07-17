# 06. モデルのエクスポート

学習済みモデル (`.ckpt` + `.pth`) を、配布・アプリ組み込み向けに変換する手順。
GPT-SoVITS には **TorchScript 版** と **ONNX 版** の 2 系統がある。

> 前提: 通常の推論は `.ckpt` + `.pth` + 事前学習モデル + ランタイムが必要
> ([05-inference 関連 / README 参照])。エクスポートはこれを**自己完結した形にまとめる**ためのもの。

---

## 方式の比較

| 方式 | スクリプト | 出力 | 用途 |
|---|---|---|---|
| **TorchScript** | `export_torch_script.py` | `.pt` 群 | Python/C++ (libtorch) で読み込み。推奨 |
| **ONNX** | `onnx_export.py` | `.onnx` 群 + json | ONNXRuntime / 多言語・軽量ランタイム |

今回のモデルは **v2Pro**。TorchScript 側は `--version` で分岐 (`export_prov2` が呼ばれる)。

---

## A. TorchScript エクスポート (推奨)

### コマンド

```bash
cd ~/projects/GPT-SoVITS && source .venv/bin/activate
NV=$PWD/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib | tr '\n' ':')$LD_LIBRARY_PATH"

# 参照テキストはファイルで渡す
echo "第12回第1項ログイン処理その1について講義します" > /tmp/ref.txt

python GPT_SoVITS/export_torch_script.py \
  --gpt_model    GPT_weights_v2Pro/yoo-e15.ckpt \
  --sovits_model SoVITS_weights_v2Pro/yoo_e8_s200.pth \
  --ref_audio    "output/slicer_opt/login_1.wav_0000021760_0000177280.wav" \
  --ref_text     /tmp/ref.txt \
  --output_path  exported/yoo \
  --version      v2Pro \
  --export_common_model \
  --device       cuda
```

### 主な引数

| 引数 | 必須 | 内容 |
|---|---|---|
| `--gpt_model` | ○ | GPT モデル (`.ckpt`) |
| `--sovits_model` | ○ | SoVITS モデル (`.pth`) |
| `--ref_audio` | ○ | 参照音声 |
| `--ref_text` | ○ | 参照テキストの**ファイルパス** (文字列でなくファイル) |
| `--output_path` | ○ | 出力ディレクトリ |
| `--version` | | モデルバージョン (既定 `v2`。今回 `v2Pro`) |
| `--export_common_model` | | BERT / SSL(HuBERT) も一緒に出力する |
| `--device` | | `cuda` / `cpu` |
| `--no-half` | | 半精度を使わない |

### 出力ファイル (`output_path` 配下)

| ファイル | 内容 | いつ出る |
|---|---|---|
| `gpt_sovits_model.pt` | GPT+SoVITS を統合した TorchScript | 常に |
| `bert_model.pt` | テキスト用 BERT | `--export_common_model` 時 |
| `ssl_model.pt` | 音声用 SSL (HuBERT) | `--export_common_model` 時 |

> `--export_common_model` を付ければ BERT/SSL も同梱でき、事前学習モデルへの
> 依存を減らせる。付けない場合は推論時に事前学習モデルが別途必要。

---

## B. ONNX エクスポート

`onnx_export.py` は現状 **スクリプト末尾にパスを直書き**する形 (CLI 引数なし)。
末尾の `__main__` を自分のモデルに書き換えて実行する。

```python
# GPT_SoVITS/onnx_export.py の __main__ を編集
if __name__ == "__main__":
    gpt_path  = "GPT_weights_v2Pro/yoo-e15.ckpt"
    vits_path = "SoVITS_weights_v2Pro/yoo_e8_s200.pth"
    exp_path  = "yoo"                 # onnx/yoo/ 配下に出力される
    export(vits_path, gpt_path, exp_path)
```

```bash
python GPT_SoVITS/onnx_export.py
```

### 出力ファイル (`onnx/yoo/` 配下)

| ファイル | 内容 |
|---|---|
| `yoo_t2s_encoder.onnx` | Text2Semantic エンコーダ |
| `yoo_t2s_fsdec.onnx` | 初回デコーダ (first-stage) |
| `yoo_t2s_sdec.onnx` | ステップデコーダ |
| `yoo_vits.onnx` | VITS (音声生成) |
| `yoo.json` | シンボル・設定 (MoeVSConf) |

> ONNX 版は分割された複数モデル + 設定 json で構成される。
> 軽量ランタイムや多言語環境 (C#, JS 等) で動かす場合に向く。

---

## 注意点

- エクスポートも torchcodec / CUDA ライブラリを使うため、
  **`LD_LIBRARY_PATH` を通した状態**で実行する ([01](01-environment.md) / [05](05-troubleshooting.md))。
- `--ref_text` は**文字列ではなくファイルパス**を渡す (よくある間違い)。
- v2Pro 特有の話者特徴を含むため、エクスポート時も `--version v2Pro` を明示する。
- 出力した `.pt` / `.onnx` を別環境で使う場合、その環境にも同じ推論ロジック
  (GPT-SoVITS の推論コード or 自作ローダ) が必要。モデル単体では動かない。
