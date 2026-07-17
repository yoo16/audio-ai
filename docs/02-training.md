# 02. 学習手順 (WebUI)

WebUI (http://localhost:9874) 上で「前処理 → データ整形 → SoVITS学習 → GPT学習」を行う。

## 前処理タブ

| ツール | 内容 | 今回の対応 |
|---|---|---|
| 0a UVR5 | BGM / 残響除去 | スキップ (BGM なし) |
| 0b 音声分割 | 長い音声を数秒単位にスライス | 実行 |
| 0c 音声認識 | Faster Whisper で文字起こし → `.list` | 実行 |
| 0d テキスト校正 | 誤認識を目視修正 | 実行 |

### 0b 音声分割
- 入力パス: `/home/yoo/projects/audio-ai/audio_converted`
- 出力: `output/slicer_opt`
- 閾値ほかパラメータはデフォルト (threshold -34 / min_length 4000 / ...)
- → 16 ファイルが 19 セグメントに

### 0c 音声認識
- 入力: `output/slicer_opt`
- 出力: `output/asr_opt/slicer_opt.list`
- ASRモデル: Faster Whisper / large-v3 / 言語 ja / float16
- ※ VAD に onnxruntime が必要 (未導入だと list が空になる → [05](05-troubleshooting.md))

### 0d テキスト校正
- listパス: `output/asr_opt/slicer_opt.list`
- 誤認識を修正して保存 (例: 「作戦」→「作成」)

---

## GPT-SoVITS TTS タブ

### 実験設定
- 実験 / モデル名: `yoo`
- 学習済みモデルバージョン: `v2Pro`

### 1A データセット整形

以下を入力:
- listパス: `output/asr_opt/slicer_opt.list`
- 音声フォルダ: `output/slicer_opt`

**ワンクリック三連** (または個別に 1Aa → 1Ab → 1Ac) を実行。生成物 (`logs/yoo/`):

| フォルダ / ファイル | 内容 | 必要なバージョン |
|---|---|---|
| `2-name2text.txt` | 音素 | 全 |
| `3-bert` | BERT 特徴 | 全 |
| `4-cnhubert` | HuBERT (SSL) 特徴 | 全 |
| `5-wav32k` | 32kHz 音声 | 全 |
| `6-name2semantic.tsv` | セマンティックトークン | 全 |
| `7-sv_cn` | 話者特徴 (SV) | **v2Pro / v2ProPlus のみ** |

> ⚠️ **v2Pro は `7-sv_cn` が別ステップ**。ワンクリック三連には含まれないため、
> 別途 SV 抽出を実行しないと SoVITS 学習が `ZeroDivisionError` で落ちる。
> 手動実行する場合:
>
> ```bash
> cd ~/projects/GPT-SoVITS && source .venv/bin/activate
> NV=$PWD/.venv/lib/python3.12/site-packages/nvidia
> export LD_LIBRARY_PATH="$(ls -d $NV/*/lib | tr '\n' ':')$LD_LIBRARY_PATH"
> export inp_text="output/asr_opt/slicer_opt.list" inp_wav_dir="output/slicer_opt" \
>        exp_name="yoo" opt_dir="logs/yoo" \
>        sv_path="GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt" \
>        i_part="0" all_parts="1" _CUDA_VISIBLE_DEVICES="0" is_half="True"
> python -s GPT_SoVITS/prepare_datasets/2-get-sv.py
> ```

### 1B SoVITS 学習 (s2)

RTX 5060 / 8GB 向け設定:

| 項目 | 値 |
|---|---|
| batch_size | 3 (OOM 時は 2→1) |
| total_epoch | 8〜10 |
| save_every_epoch | 4 |
| fp16 | ON |

- 話者特徴 (7-sv_cn) を使う。**7-sv_cn が揃っていることが前提**。
- 出力: `SoVITS_weights_v2Pro/yoo_e8_s200.pth`
- 学習時 VRAM 約 7.2GB / 使用率 85% で完走。

### 1B GPT 学習 (s1)

| 項目 | 値 |
|---|---|
| batch_size | 2〜3 |
| total_epoch | 15 |
| save_every_epoch | 5 |
| fp16 | ON |

- **音素 (2-name2text) とセマンティック (6-name2semantic) のみ使用**。話者特徴 (7-sv_cn) は不要。
  → SV 抽出前でも GPT 学習だけは成功する。
- torch>=2.6 の `weights_only=True` 問題で ckpt ロードが落ちるため、`s1_train.py` にパッチ必要 ([05](05-troubleshooting.md))。
- 出力: `GPT_weights_v2Pro/yoo-e15.ckpt`

---

## 完成物

| モデル | ファイル | 学習に使うデータ |
|---|---|---|
| GPT (s1) | `GPT_weights_v2Pro/yoo-e15.ckpt` | 音素 + セマンティック |
| SoVITS (s2) | `SoVITS_weights_v2Pro/yoo_e8_s200.pth` | 音素 + SSL + wav + 話者特徴 |

両方揃ったら [03-inference.md](03-inference.md) へ。
