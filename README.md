# audio-ai

自分の音声から、任意テキストを自分の声で読み上げる音声合成モデルを作るプロジェクト。
文字起こし (faster-whisper) → GPT-SoVITS 学習 → 推論 の一連を扱う。

## 構成

| ファイル / ディレクトリ | 内容 |
|---|---|
| `stt.py` | faster-whisper で `audio/` を一括文字起こし → `transcriptions.list` |
| `convert.py` | WAV を GPT-SoVITS 用 (16bit PCM / 32kHz / mono) に変換 |
| `audio/` | 元音声 (学習素材) |
| `audio_converted/` | 変換後の音声 |
| `transcriptions.list` | 文字起こし結果 (`パス\|話者\|言語\|テキスト`) |

## ドキュメント (用途別)

| ドキュメント | 用途 |
|---|---|
| [docs/01-environment.md](docs/01-environment.md) | 環境構築 (stt.py / GPT-SoVITS のインストール) |
| [docs/02-training.md](docs/02-training.md) | WebUI での学習手順 (前処理〜SoVITS/GPT学習) |
| [docs/03-inference.md](docs/03-inference.md) | 推論と品質チューニング |
| [docs/04-dataset.md](docs/04-dataset.md) | 学習データの必要量・録音の指針 |
| [docs/05-troubleshooting.md](docs/05-troubleshooting.md) | エラーと対処 (ハマりどころ集) |

## 環境

- WSL2 (Ubuntu) / Python 3.12
- GPU: RTX 5060 (Blackwell / sm_120, VRAM 8GB)
- PyTorch cu128 (CUDA 12.8)
- パッケージ管理: uv

## クイックフロー

```
1. audio/ に音声を用意 (合計5〜10分推奨)
2. python stt.py            # 文字起こし
3. python convert.py        # 16bit PCM / 32kHz へ変換
4. GPT-SoVITS WebUI で学習   # docs/02 参照
5. 推論で自分の声を生成       # docs/03 参照
```
